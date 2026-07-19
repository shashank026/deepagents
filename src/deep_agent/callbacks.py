"""Callbacks for streaming agent execution updates to CLI."""
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

from cli import AgentStatusStreamer


class StreamingCLICallback(BaseCallbackHandler):
    """Callback handler that streams agent progress to CLI with colors."""
    
    def __init__(self):
        super().__init__()
        self.streamer = AgentStatusStreamer()
        self.current_tool = None
        self.tool_count = 0
        self.tool_call_counts = {}  # Track calls per tool
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.streamer.info("🧠 Agent thinking...")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        pass  # Don't clutter output
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.streamer.error(f"LLM error: {str(error)}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        self.tool_count += 1
        tool_name = serialized.get("name", "unknown")
        self.current_tool = tool_name
        
        # Track individual tool call counts
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1
        call_num = self.tool_call_counts[tool_name]
        
        # Map tool names to friendly descriptions with dynamic context
        tool_descriptions = {
            "list_tables": f"📋 Discovering database tables",
            "get_table": f"🔍 Analyzing table schema (call #{call_num})",
            "search_tables": f"🔎 Searching for relevant tables (search #{call_num})",
            "run_safe_read_query": f"⚡ Executing query #{call_num}",
        }
        
        # Add input context for queries
        if tool_name == "run_safe_read_query" and input_str:
            # Extract table name from query if possible
            import re
            table_match = re.search(r'FROM\s+(\w+)', input_str, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                description = f"⚡ Querying {table_name} (query #{call_num})"
            else:
                description = tool_descriptions.get(tool_name, f"🔧 Using {tool_name}")
        elif tool_name == "get_table" and input_str:
            description = f"🔍 Examining {input_str} schema"
        elif tool_name == "search_tables" and input_str:
            description = f"🔎 Searching for '{input_str}'"
        else:
            description = tool_descriptions.get(tool_name, f"🔧 Using {tool_name} (call #{call_num})")
        
        self.streamer.update_status(f"{description}...", style="yellow")
    
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        if self.current_tool:
            # Show brief summary of what was found
            if self.current_tool == "run_safe_read_query":
                # Try to count results
                import json
                try:
                    if output.strip().startswith('['):
                        results = json.loads(output)
                        count = len(results)
                        self.streamer.update_status(
                            f"✓ Found {count} record(s)", 
                            style="green"
                        )
                    else:
                        self.streamer.update_status(
                            f"✓ Query completed", 
                            style="green"
                        )
                except:
                    self.streamer.update_status(
                        f"✓ Query completed", 
                        style="green"
                    )
            else:
                self.streamer.update_status(
                    f"✓ Completed", 
                    style="green"
                )
        self.current_tool = None
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when tool errors."""
        self.streamer.error(f"Tool error: {str(error)}")
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.streamer.info(f"🎯 Action: {action.tool}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        self.streamer.success("🎉 Investigation complete!")
    
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        pass  # Don't clutter output
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        pass  # Don't clutter output


class InvestigationStageCallback(BaseCallbackHandler):
    """Callback that tracks investigation stages."""
    
    STAGE_MARKERS = {
        "understanding": "🔍 Understanding your question",
        "schema_exploration": "📚 Exploring database schema",
        "evidence_collection": "🔎 Collecting evidence",
        "query_execution": "⚡ Executing queries",
        "analysis": "📊 Analyzing results",
        "conclusion": "📝 Generating insights"
    }
    
    def __init__(self):
        super().__init__()
        self.streamer = AgentStatusStreamer()
        self.current_stage = None
        self.stages_completed = set()
    
    def enter_stage(self, stage: str):
        """Enter a new investigation stage."""
        if stage in self.STAGE_MARKERS and stage not in self.stages_completed:
            self.current_stage = stage
            self.streamer.print_stage(self.STAGE_MARKERS[stage], "in_progress")
    
    def complete_stage(self, stage: str):
        """Mark a stage as completed."""
        if stage in self.STAGE_MARKERS:
            self.stages_completed.add(stage)
            self.streamer.print_stage(self.STAGE_MARKERS[stage], "completed")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Detect stage based on tool usage."""
        tool_name = serialized.get("name", "")
        
        if tool_name == "list_tables":
            self.enter_stage("schema_exploration")
        elif tool_name in ["search_tables", "get_table"]:
            self.enter_stage("evidence_collection")
        elif tool_name == "execute_query":
            self.enter_stage("query_execution")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Complete stage after tool execution."""
        if self.current_stage:
            self.complete_stage(self.current_stage)
