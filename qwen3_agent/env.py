"""Gymnasium environment for email search agent training with SB3."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import asdict

from qwen3_agent.data.types import SyntheticQuery
from qwen3_agent.tools import search_emails, read_email
from qwen3_agent.evaluation import determine_if_answer_is_correct, EvaluationRubric


class EmailSearchEnv(gym.Env):
    """Email search environment for RL training.
    
    This environment wraps the email search task as a Gym environment suitable
    for use with Stable Baselines 3. The agent receives text observations and
    must select actions (tool calls) to search and read emails to answer queries.
    
    Observation Space:
        The observation is a dictionary containing:
        - "text": Current conversation history as a string
        - "turn": Current turn number
        - "done": Whether the episode is done
    
    Action Space:
        The action is a string representing a JSON tool call:
        {
            "tool_name": "search_emails" | "read_email" | "return_final_answer",
            "tool_args": {...}
        }
        
        In practice, with SB3, we'll use a custom policy that generates these strings.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        scenarios: List[SyntheticQuery],
        max_turns: int = 10,
        verbose: bool = False,
    ):
        """Initialize the environment.
        
        Args:
            scenarios: List of synthetic queries to train on
            max_turns: Maximum number of turns per episode
            verbose: Whether to print detailed logs
        """
        super().__init__()
        
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.verbose = verbose
        
        # Current episode state
        self.current_scenario: Optional[SyntheticQuery] = None
        self.current_turn = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.rubric = EvaluationRubric()
        self.done = False
        
        # Define observation and action spaces
        # For text-based RL, we use Box with dummy dimensions
        # The actual processing will be done by the policy
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=50000),
            "turn": spaces.Box(low=0, high=max_turns, shape=(1,), dtype=np.int32),
            "done": spaces.Discrete(2),
        })
        
        # Action space is continuous for compatibility with SB3
        # But we'll interpret it as discrete tool selection
        self.action_space = spaces.Discrete(3)  # 3 tools: search, read, answer
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'scenario_idx')
        
        Returns:
            observation: Initial observation
            info: Additional info dictionary
        """
        super().reset(seed=seed)
        
        # Select a random scenario or use specified index
        if options and "scenario_idx" in options:
            scenario_idx = options["scenario_idx"]
        else:
            scenario_idx = self.np_random.integers(0, len(self.scenarios))
        
        self.current_scenario = self.scenarios[scenario_idx]
        self.current_turn = 0
        self.done = False
        self.rubric = EvaluationRubric()
        
        # Initialize conversation with system prompt and user query
        system_prompt = (
            f"You are an email search agent. You are given a user query and a list of tools "
            f"you can use to search the user's email. Use the tools to search the user's emails "
            f"and find the answer to the user's query. You may take up to {self.max_turns} turns "
            f"to find the answer, so if your first search doesn't find the answer, you can try "
            f"with different keywords.\n\n"
            f"User's email address is {self.current_scenario.inbox_address}\n"
            f"Today's date is {self.current_scenario.query_date}\n\n"
            f"Available tools:\n"
            f"1. search_emails(keywords: List[str], from_addr: str = None, to_addr: str = None, "
            f"sent_after: str = None, sent_before: str = None, max_results: int = 10)\n"
            f"2. read_email(message_id: str)\n"
            f"3. return_final_answer(answer: str, sources: List[str])"
        )
        
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.current_scenario.question},
        ]
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"New Episode - Scenario {scenario_idx}")
            print(f"Question: {self.current_scenario.question}")
            print(f"{'='*60}")
        
        return observation, info
    
    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Dictionary with 'tool_name' and 'tool_args'
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is done (terminal state)
            truncated: Whether episode was truncated (max turns)
            info: Additional info
        """
        if self.done:
            raise RuntimeError("Episode is done, call reset() first")
        
        self.current_turn += 1
        self.rubric.num_turns = self.current_turn
        
        # Parse action
        tool_name = action.get("tool_name")
        tool_args = action.get("tool_args", {})
        
        if self.verbose:
            print(f"\nTurn {self.current_turn}/{self.max_turns}")
            print(f"Tool: {tool_name}")
            print(f"Args: {tool_args}")
        
        # Add assistant message
        self.conversation_history.append({
            "role": "assistant",
            "content": json.dumps({"tool_name": tool_name, "tool_args": tool_args})
        })
        
        # Execute tool and calculate reward
        reward = 0.0
        terminated = False
        truncated = False
        
        try:
            # Execute the tool
            if tool_name == "search_emails":
                result = self._execute_search(tool_args)
            elif tool_name == "read_email":
                result = self._execute_read(tool_args)
            elif tool_name == "return_final_answer":
                reward, terminated = self._execute_answer(tool_args)
                result = {"status": "answer_provided"}
            else:
                self.rubric.bad_tool_call_name = True
                result = {"error": f"Unknown tool: {tool_name}"}
                reward = -2.0
                terminated = True
        except Exception as e:
            self.rubric.bad_tool_call_args = True
            result = {"error": str(e)}
            reward = -1.8
            terminated = True
        
        # Add tool result to conversation
        self.conversation_history.append({
            "role": "tool",
            "content": json.dumps(result)
        })
        
        # Check if max turns reached
        if self.current_turn >= self.max_turns and not terminated:
            self.rubric.ran_out_of_turns = True
            truncated = True
            reward = 0.0  # Partial reward for running out of turns
        
        self.done = terminated or truncated
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.verbose and self.done:
            print(f"\nEpisode finished")
            print(f"Reward: {reward:.2f}")
            print(f"Answer correct: {self.rubric.answer_correct}")
        
        return observation, reward, terminated, truncated, info
    
    def _execute_search(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_emails tool."""
        if "keywords" not in tool_args or not isinstance(tool_args["keywords"], list):
            raise ValueError("search_emails requires 'keywords' as a list")
        
        search_results = search_emails(
            inbox=self.current_scenario.inbox_address,
            keywords=tool_args["keywords"],
            from_addr=tool_args.get("from_addr"),
            to_addr=tool_args.get("to_addr"),
            sent_after=tool_args.get("sent_after"),
            sent_before=tool_args.get("sent_before"),
            max_results=tool_args.get("max_results", 10),
        )
        
        # Check if we found the right email
        for result in search_results:
            if result.message_id == self.current_scenario.message_ids[0]:
                self.rubric.ever_found_right_email = True
                break
        
        if self.verbose:
            print(f"Found {len(search_results)} emails")
        
        return {"results": [asdict(r) for r in search_results]}
    
    def _execute_read(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute read_email tool."""
        message_id = tool_args.get("message_id")
        if not isinstance(message_id, str):
            raise ValueError("read_email requires 'message_id' as a string")
        
        # Check if reading the right email
        if message_id == self.current_scenario.message_ids[0]:
            self.rubric.ever_read_right_email = True
        
        email = read_email(message_id)
        if email is None:
            self.rubric.ever_tried_to_read_invalid_email = True
            return {"error": "Email not found"}
        
        if self.verbose:
            print(f"Read email: {email.subject}")
        
        return email.model_dump()
    
    def _execute_answer(self, tool_args: Dict[str, Any]) -> Tuple[float, bool]:
        """Execute return_final_answer tool and calculate reward."""
        answer = tool_args.get("answer")
        sources = tool_args.get("sources", [])
        
        if answer is None or not isinstance(sources, list):
            self.rubric.bad_tool_call_args = True
            return -1.8, True
        
        self.rubric.num_sources = len(sources)
        
        if answer == "I don't know":
            self.rubric.returned_i_dont_know = True
            reward = 0.0 + self._calculate_partial_rewards()
        else:
            self.rubric.attempted_answer = True
            # Use synchronous version for now (we'll handle async in training loop)
            # For now, use a simple heuristic
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self.rubric.answer_correct = loop.run_until_complete(
                determine_if_answer_is_correct(
                    answer, self.current_scenario, verbose=self.verbose
                )
            )
            self.rubric.sources_correct = self.current_scenario.message_ids[0] in sources
            
            reward = self._calculate_reward()
        
        if self.verbose:
            print(f"Answer correct: {self.rubric.answer_correct}")
            print(f"Sources correct: {self.rubric.sources_correct}")
        
        return reward, True
    
    def _calculate_partial_rewards(self) -> float:
        """Calculate partial rewards for intermediate progress."""
        partial = 0.0
        partial += 0.1 if self.rubric.ever_found_right_email else 0
        partial += 0.1 if self.rubric.ever_read_right_email else 0
        partial += 0.1 if not self.rubric.ever_tried_to_read_invalid_email else 0
        partial += 0.1 if self.rubric.sources_correct else 0
        return partial
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on rubric."""
        partial_rewards = self._calculate_partial_rewards()
        
        # Formatting errors: -2 to -1
        if self.rubric.cant_parse_tool_call:
            return -2 + partial_rewards
        if self.rubric.bad_tool_call_name:
            return -1.9 + partial_rewards
        if self.rubric.bad_tool_call_args:
            return -1.8 + partial_rewards
        
        # Wrong answer: -1 to 0
        if self.rubric.attempted_answer and not self.rubric.answer_correct:
            return -1 + partial_rewards
        
        # No answer: 0 to 1
        if self.rubric.returned_i_dont_know or self.rubric.ran_out_of_turns:
            return 0 + partial_rewards
        
        # Correct answer: 1 to 2
        if self.rubric.answer_correct:
            reward = 1.0
            reward += 0.3 if self.rubric.sources_correct else 0
            reward += 0.1 / max(self.rubric.num_sources, 1)
            reward += 0.1 * (1 - self.current_turn / self.max_turns)
            return reward
        
        return 0.0
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        # Convert conversation history to text
        text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.conversation_history
        ])
        
        return {
            "text": text,
            "turn": np.array([self.current_turn], dtype=np.int32),
            "done": 1 if self.done else 0,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        return {
            "rubric": asdict(self.rubric),
            "scenario_id": self.current_scenario.id if self.current_scenario else None,
            "conversation_length": len(self.conversation_history),
        }
    
    def render(self):
        """Render the environment (human-readable)."""
        if self.current_scenario is None:
            print("Environment not initialized")
            return
        
        print(f"\n{'='*60}")
        print(f"Scenario: {self.current_scenario.question}")
        print(f"Turn: {self.current_turn}/{self.max_turns}")
        print(f"{'='*60}")
        for msg in self.conversation_history[-4:]:  # Show last 4 messages
            print(f"{msg['role'].upper()}: {msg['content'][:200]}...")
        print(f"{'='*60}\n")
    
    def close(self):
        """Clean up resources."""
        pass

