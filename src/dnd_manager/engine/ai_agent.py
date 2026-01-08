"""AI Agent for autonomous combatant decision-making.

This module implements the DungeonAgent class that makes tactical decisions
for AI-controlled combatants based on their PersonaComponent, current game
state, and available actions.

The agent uses LLM-based reasoning with tool calling to select and execute
appropriate combat actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dnd_manager.core.config import get_settings
from dnd_manager.core.exceptions import AIControlError, AIResponseError
from dnd_manager.core.logging import get_logger
from dnd_manager.engine.tools import (
    ToolCall,
    ToolResult,
    execute_tool_calls,
    get_tools_as_openai_schema,
)
from dnd_manager.models import AutonomyLevel


if TYPE_CHECKING:
    from dnd_manager.models import Combatant
    from dnd_manager.models.game_state import Scene

logger = get_logger(__name__)


# =============================================================================
# Context Building
# =============================================================================


@dataclass
class BattlefieldContext:
    """Summarized battlefield state for AI decision-making.

    Attributes:
        combatant_name: Name of the acting combatant.
        combatant_hp: Current/max HP.
        combatant_ac: Armor class.
        allies: List of ally descriptions.
        enemies: List of enemy descriptions.
        environment: Environmental notes.
        round_number: Current combat round.
        special_conditions: Any active conditions or effects.
    """

    combatant_name: str
    combatant_hp: str
    combatant_ac: int
    combatant_position: str
    allies: list[str] = field(default_factory=list)
    enemies: list[str] = field(default_factory=list)
    environment: str = ""
    round_number: int = 1
    special_conditions: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Convert to text suitable for LLM prompt."""
        lines = [
            f"## Current Battlefield (Round {self.round_number})",
            "",
            f"**You:** {self.combatant_name}",
            f"- HP: {self.combatant_hp}",
            f"- AC: {self.combatant_ac}",
            f"- Position: {self.combatant_position}",
        ]

        if self.special_conditions:
            lines.append(f"- Conditions: {', '.join(self.special_conditions)}")

        lines.append("")

        if self.allies:
            lines.append("**Allies:**")
            for ally in self.allies:
                lines.append(f"- {ally}")
            lines.append("")

        if self.enemies:
            lines.append("**Enemies:**")
            for enemy in self.enemies:
                lines.append(f"- {enemy}")
            lines.append("")

        if self.environment:
            lines.append(f"**Environment:** {self.environment}")

        return "\n".join(lines)


def build_battlefield_context(
    combatant: "Combatant",
    scene: "Scene",
) -> BattlefieldContext:
    """Build battlefield context from game state.

    Args:
        combatant: The acting combatant.
        scene: Current scene with all combatants.

    Returns:
        BattlefieldContext for AI prompting.
    """
    from dnd_manager.models import Monster, NPC, PlayerCharacter
    from dnd_manager.models.enums import CombatantType

    # Determine allies and enemies based on combatant type
    allies: list[str] = []
    enemies: list[str] = []

    for other in scene.combatants:
        if other.uid == combatant.uid:
            continue

        # Build description
        hp_status = "healthy"
        hp_pct = other.health.hp_percentage
        if hp_pct < 25:
            hp_status = "badly wounded"
        elif hp_pct < 50:
            hp_status = "wounded"
        elif hp_pct < 75:
            hp_status = "lightly wounded"

        conditions = [c.value for c in other.health.conditions]
        condition_str = f" ({', '.join(conditions)})" if conditions else ""

        desc = f"{other.name} ({hp_status}, AC {other.armor_class}){condition_str}"

        # Categorize as ally or enemy
        if combatant.type == CombatantType.MONSTER:
            # Monsters see PCs as enemies, other monsters as allies
            if isinstance(other, (PlayerCharacter,)):
                enemies.append(desc)
            elif isinstance(other, Monster):
                allies.append(desc)
            elif isinstance(other, NPC):
                if other.attitude in ("hostile", "unfriendly"):
                    allies.append(desc)  # NPC hostile to party = ally to monster
                else:
                    enemies.append(desc)
        else:
            # PCs/NPCs see monsters as enemies
            if isinstance(other, Monster):
                enemies.append(desc)
            elif isinstance(other, PlayerCharacter):
                allies.append(desc)
            elif isinstance(other, NPC):
                if other.attitude in ("friendly", "helpful"):
                    allies.append(desc)
                elif other.attitude in ("hostile", "unfriendly"):
                    enemies.append(desc)

    # Build HP string
    hp_str = f"{combatant.health.current_hp}/{combatant.health.max_hp}"
    if combatant.health.temp_hp > 0:
        hp_str += f" (+{combatant.health.temp_hp} temp)"

    # Conditions
    conditions = [c.value for c in combatant.health.conditions]

    return BattlefieldContext(
        combatant_name=combatant.name,
        combatant_hp=hp_str,
        combatant_ac=combatant.armor_class,
        combatant_position="in combat",  # Could be enhanced with actual positioning
        allies=allies,
        enemies=enemies,
        environment=scene.terrain_description or scene.description[:200],
        round_number=scene.turn_order.current_round,
        special_conditions=conditions,
    )


# =============================================================================
# System Prompts
# =============================================================================


DUNGEON_AGENT_SYSTEM_PROMPT = """You are an AI controlling a character in a D&D 5E combat encounter. You must decide what action to take on your turn based on your character's personality, goals, and the current battlefield situation.

## Your Character
{persona_context}

## Combat Rules
- You have ONE action, ONE bonus action, and movement each turn
- You can also use ONE reaction (if triggered) before your next turn
- Choose actions that align with your character's personality and directives
- Consider tactical positioning, ally protection, and threat assessment

## Decision Guidelines
Based on your personality and directives:
{directive_guidelines}

## Available Actions
Use the provided tools to take actions. Common choices:
- `roll_attack`: Make a melee or ranged attack
- `cast_spell`: Cast a spell (if you can cast spells)
- `move`: Move to a new position
- `dash`: Double your movement this turn
- `dodge`: Make yourself harder to hit
- `disengage`: Move without provoking opportunity attacks
- `help_action`: Give an ally advantage
- `hide`: Attempt to become hidden
- `use_item`: Use an item from inventory

## Response Format
1. First, briefly reason about the situation (1-2 sentences)
2. Then, call the appropriate tool(s) to execute your action
3. If you need to attack, you MUST call roll_attack with appropriate parameters

Think tactically but stay in character!"""


def build_persona_context(combatant: "Combatant") -> str:
    """Build persona context for the system prompt.

    Args:
        combatant: The acting combatant.

    Returns:
        Persona context string.
    """
    persona = combatant.persona
    lines = [
        f"**Name:** {persona.name}",
    ]

    if persona.biography:
        # Truncate long biographies
        bio = persona.biography[:500]
        if len(persona.biography) > 500:
            bio += "..."
        lines.append(f"**Background:** {bio}")

    if persona.personality_traits:
        lines.append(f"**Personality:** {', '.join(persona.personality_traits)}")

    if persona.voice_style and persona.voice_style != "Natural":
        lines.append(f"**Voice/Manner:** {persona.voice_style}")

    if persona.flaws:
        lines.append(f"**Flaws:** {', '.join(persona.flaws)}")

    return "\n".join(lines)


def build_directive_guidelines(combatant: "Combatant") -> str:
    """Build directive guidelines from persona.

    Args:
        combatant: The acting combatant.

    Returns:
        Directive guidelines string.
    """
    persona = combatant.persona
    guidelines: list[str] = []

    # Add explicit directives
    for directive in persona.directives:
        guidelines.append(f"- {directive}")

    # Infer guidelines from personality traits
    traits_lower = [t.lower() for t in persona.personality_traits]
    flaws_lower = [f.lower() for f in persona.flaws]

    if any("coward" in t or "fearful" in t for t in traits_lower + flaws_lower):
        guidelines.append("- You prefer to avoid direct confrontation; consider retreating or hiding")

    if any("protect" in t or "loyal" in t for t in traits_lower):
        guidelines.append("- Prioritize protecting allies over dealing damage")

    if any("aggress" in t or "violent" in t or "bloodthirst" in t for t in traits_lower):
        guidelines.append("- You prefer direct combat and attacking the nearest enemy")

    if any("clever" in t or "tactical" in t or "cunning" in t for t in traits_lower):
        guidelines.append("- Use positioning and tactics; target vulnerable enemies")

    if not guidelines:
        guidelines.append("- Act according to your character's nature")
        guidelines.append("- Prioritize survival while accomplishing objectives")

    return "\n".join(guidelines)


# =============================================================================
# AI Agent
# =============================================================================


@dataclass
class AgentDecision:
    """Result of an AI agent decision.

    Attributes:
        reasoning: The agent's reasoning for its action.
        tool_calls: List of tool calls to execute.
        raw_response: Raw LLM response for debugging.
    """

    reasoning: str
    tool_calls: list[ToolCall]
    raw_response: str = ""


class DungeonAgent:
    """AI Agent for making combat decisions for autonomous combatants.

    Uses LLM with tool calling to select and execute appropriate
    combat actions based on persona and battlefield state.

    Attributes:
        model: LLM model to use.
        temperature: Sampling temperature for variety.
        max_retries: Maximum retry attempts on failure.
    """

    def __init__(
        self,
        *,
        model: str = "google/gemini-2.0-flash-001",
        temperature: float = 0.7,
        max_retries: int = 2,
        use_openrouter: bool = True,
    ) -> None:
        """Initialize the DungeonAgent.

        Args:
            model: LLM model identifier.
            temperature: Sampling temperature (higher = more creative).
            max_retries: Maximum retry attempts.
            use_openrouter: Whether to use OpenRouter API.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.use_openrouter = use_openrouter
        self._client: Any = None

        logger.info(
            "DungeonAgent initialized",
            model=model,
            temperature=temperature,
        )

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                settings = get_settings()

                if self.use_openrouter:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={
                            "HTTP-Referer": "https://github.com/dnd-campaign-manager",
                            "X-Title": "D&D Campaign Manager",
                        },
                    )
                else:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(api_key=api_key)

            except ImportError as exc:
                raise AIControlError(
                    "openai package not installed",
                    provider="openrouter" if self.use_openrouter else "openai",
                ) from exc

        return self._client

    def _parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Parse tool calls from LLM response.

        Args:
            response: OpenAI chat completion response.

        Returns:
            List of ToolCall objects.
        """
        tool_calls: list[ToolCall] = []
        message = response.choices[0].message

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(ToolCall(
                    tool_name=tc.function.name,
                    arguments=arguments,
                    call_id=tc.id,
                ))

        return tool_calls

    def decide_action(
        self,
        combatant: "Combatant",
        scene: "Scene",
    ) -> AgentDecision:
        """Decide what action to take for a combatant.

        Args:
            combatant: The acting combatant.
            scene: Current scene with battlefield state.

        Returns:
            AgentDecision with reasoning and tool calls.

        Raises:
            AIControlError: If decision-making fails.
        """
        # Build context
        battlefield = build_battlefield_context(combatant, scene)
        persona_context = build_persona_context(combatant)
        directive_guidelines = build_directive_guidelines(combatant)

        # Build system prompt
        system_prompt = DUNGEON_AGENT_SYSTEM_PROMPT.format(
            persona_context=persona_context,
            directive_guidelines=directive_guidelines,
        )

        # Build user message with battlefield state
        user_message = f"""It's your turn to act!

{battlefield.to_prompt_text()}

What do you do? Consider your personality, directives, and the tactical situation.
Call the appropriate tool(s) to execute your action."""

        # Get tools schema
        tools = get_tools_as_openai_schema()

        logger.debug(
            "Requesting AI decision",
            combatant=combatant.name,
            model=self.model,
        )

        for attempt in range(self.max_retries + 1):
            try:
                from openai import APIConnectionError, APIStatusError, RateLimitError

                client = self._get_client()

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=1024,
                )

                # Extract reasoning and tool calls
                message = response.choices[0].message
                reasoning = message.content or ""
                tool_calls = self._parse_tool_calls(response)

                # If no tool calls, try to extract action from text
                if not tool_calls and reasoning:
                    logger.warning(
                        "No tool calls in response, agent may not have acted",
                        combatant=combatant.name,
                    )

                logger.info(
                    "AI decision made",
                    combatant=combatant.name,
                    tool_calls=len(tool_calls),
                    reasoning_preview=reasoning[:100] if reasoning else "",
                )

                return AgentDecision(
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    raw_response=str(response),
                )

            except RateLimitError as exc:
                if attempt < self.max_retries:
                    logger.warning(f"Rate limited, retrying ({attempt + 1}/{self.max_retries})")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise AIControlError(
                    f"Rate limit exceeded after {self.max_retries} retries",
                    provider="openrouter" if self.use_openrouter else "openai",
                    model=self.model,
                ) from exc

            except APIConnectionError as exc:
                raise AIControlError(
                    f"Failed to connect to AI provider: {exc}",
                    provider="openrouter" if self.use_openrouter else "openai",
                    model=self.model,
                ) from exc

            except APIStatusError as exc:
                raise AIControlError(
                    f"AI API error: {exc}",
                    provider="openrouter" if self.use_openrouter else "openai",
                    model=self.model,
                    details={"status_code": exc.status_code},
                ) from exc

            except Exception as exc:
                raise AIControlError(
                    f"AI decision failed: {exc}",
                    provider="openrouter" if self.use_openrouter else "openai",
                    model=self.model,
                ) from exc

        # Should not reach here
        raise AIControlError("Decision failed after all retries", model=self.model)

    def execute_decision(
        self,
        decision: AgentDecision,
    ) -> list[ToolResult]:
        """Execute the tool calls from a decision.

        Args:
            decision: The agent's decision with tool calls.

        Returns:
            List of tool execution results.
        """
        if not decision.tool_calls:
            logger.warning("No tool calls to execute")
            return []

        results = execute_tool_calls(decision.tool_calls)

        for result in results:
            if result.success:
                logger.info(
                    "Tool executed successfully",
                    tool=result.tool_name,
                    result_preview=result.result[:100],
                )
            else:
                logger.error(
                    "Tool execution failed",
                    tool=result.tool_name,
                    error=result.error,
                )

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def get_ai_action(
    combatant: "Combatant",
    scene: "Scene",
    *,
    model: str = "google/gemini-2.0-flash-001",
) -> tuple[AgentDecision, list[ToolResult]]:
    """Get and execute an AI action for a combatant.

    Convenience function that creates an agent, gets a decision,
    and executes the resulting tool calls.

    Args:
        combatant: The acting combatant.
        scene: Current scene.
        model: LLM model to use.

    Returns:
        Tuple of (decision, results).
    """
    agent = DungeonAgent(model=model)
    decision = agent.decide_action(combatant, scene)
    results = agent.execute_decision(decision)
    return decision, results


__all__ = [
    "BattlefieldContext",
    "build_battlefield_context",
    "build_persona_context",
    "build_directive_guidelines",
    "AgentDecision",
    "DungeonAgent",
    "get_ai_action",
    "DUNGEON_AGENT_SYSTEM_PROMPT",
]
