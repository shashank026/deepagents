"""Interactive CLI with colorful streaming status updates."""
import sys
from typing import Optional, Callable
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import click

console = Console()


class AgentStatusStreamer:
    """Manages streaming status updates for the agent."""
    
    def __init__(self):
        self.console = console
        self.current_stage = None
        self.live = None
        
    @contextmanager
    def status(self, message: str, stage: str = "processing", spinner: str = "dots"):
        """Context manager for status updates with spinner."""
        try:
            with self.console.status(f"[bold cyan]{message}[/]", spinner=spinner) as status:
                self.current_stage = stage
                yield status
        finally:
            self.current_stage = None
    
    def update_status(self, message: str, style: str = "cyan"):
        """Update the current status message."""
        if self.current_stage:
            self.console.print(f"[{style}]→[/] {message}")
    
    def success(self, message: str):
        """Display success message."""
        self.console.print(f"[bold green]✓[/] {message}")
    
    def error(self, message: str):
        """Display error message."""
        self.console.print(f"[bold red]✗[/] {message}")
    
    def warning(self, message: str):
        """Display warning message."""
        self.console.print(f"[bold yellow]⚠[/] {message}")
    
    def info(self, message: str):
        """Display info message."""
        self.console.print(f"[bold blue]ℹ[/] {message}")
    
    def display_result(self, title: str, content: str):
        """Display results in a panel."""
        panel = Panel(
            content,
            title=f"[bold green]{title}[/]",
            border_style="green",
            expand=False
        )
        self.console.print(panel)
    
    def display_table(self, data: list[dict], title: str = "Results"):
        """Display data in a table format."""
        if not data:
            self.warning("No data to display")
            return
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        # Add columns from first row
        for key in data[0].keys():
            table.add_column(str(key), style="cyan")
        
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
        
        self.console.print(table)
    
    def print_stage(self, stage_name: str, status: str = "in_progress"):
        """Print a stage with appropriate styling."""
        styles = {
            "in_progress": ("cyan", "⏳"),
            "completed": ("green", "✓"),
            "failed": ("red", "✗"),
            "pending": ("dim", "○")
        }
        style, icon = styles.get(status, ("white", "•"))
        self.console.print(f"[bold {style}]{icon} {stage_name}[/]")


class InvestigationProgress:
    """Manages multi-stage investigation progress display."""
    
    STAGES = [
        ("Understanding Query", "analyzing"),
        ("Collecting Evidence", "searching"),
        ("Exploring Database", "database"),
        ("Executing Queries", "executing"),
        ("Analyzing Results", "thinking"),
        ("Generating Report", "writing")
    ]
    
    def __init__(self):
        self.console = console
        self.streamer = AgentStatusStreamer()
        self.completed_stages = []
        
    def run_stage(self, stage_index: int, callback: Optional[Callable] = None):
        """Run a specific stage with status updates."""
        if stage_index >= len(self.STAGES):
            return
        
        stage_name, spinner_type = self.STAGES[stage_index]
        
        # Map spinner types to rich spinner names
        spinner_map = {
            "analyzing": "dots",
            "searching": "line",
            "database": "dots12",
            "executing": "arc",
            "thinking": "bouncingBall",
            "writing": "dots8Bit"
        }
        
        spinner = spinner_map.get(spinner_type, "dots")
        
        with self.streamer.status(stage_name, stage=stage_name, spinner=spinner):
            if callback:
                callback()
        
        self.completed_stages.append(stage_name)
        self.streamer.success(f"{stage_name} completed")


@click.command()
@click.argument('question', required=False)
@click.option('--interactive', '-i', is_flag=True, help='Start in interactive mode')
def main(question: Optional[str], interactive: bool):
    """
    Deep Agent CLI - Intelligent Database Investigation Tool
    
    Ask questions about your database and get intelligent insights.
    """
    console.print("\n[bold cyan]🔍 Deep Agent - Database Investigation System[/]\n")
    
    if interactive or not question:
        run_interactive_mode()
    else:
        run_single_query(question)


def run_interactive_mode():
    """Run in interactive mode where user can ask multiple questions."""
    streamer = AgentStatusStreamer()
    
    console.print("[yellow]Interactive mode started. Type 'exit' or 'quit' to stop.[/]\n")
    
    while True:
        try:
            # Get user input
            question = console.input("[bold green]❓ Your question:[/] ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("\n[bold cyan]👋 Goodbye![/]\n")
                break
            
            if not question.strip():
                continue
            
            # Process the question
            process_question(question, streamer)
            console.print()  # Add spacing between queries
            
        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]⚠ Interrupted by user[/]")
            break
        except EOFError:
            break


def run_single_query(question: str):
    """Run a single query and exit."""
    streamer = AgentStatusStreamer()
    console.print(f"[bold]Question:[/] {question}\n")
    process_question(question, streamer)


def process_question(question: str, streamer: AgentStatusStreamer):
    """
    Process a user question through the investigation agent.

    Returns:
        A validated RootCauseReport when successful, otherwise None.
    """
    import time
    from agent import agent
    from callbacks import StreamingCLICallback, InvestigationStageCallback
    
    console.print()
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create callbacks for streaming updates
            cli_callback = StreamingCLICallback()
            stage_callback = InvestigationStageCallback()
            
            # Initial stage
            with streamer.status("🔍 Understanding your question...", spinner="dots"):
                stage_callback.enter_stage("understanding")
            stage_callback.complete_stage("understanding")
            
            console.print()
            streamer.print_stage("🚀 Starting Investigation", "in_progress")
            console.print()
            
            # Invoke agent with streaming callbacks
            with streamer.status("🔄 Agent working...", spinner="dots"):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={
                        "callbacks": [cli_callback, stage_callback]
                    }
                )

            # structured_response = result.get("structured_response")

            # if structured_response is None:
            #     raise ValueError(
            #         "Agent completed without returning "
            #         "'structured_response'."
            #     )

            # if isinstance(structured_response, RootCauseReport):
            #     report = structured_response
            # else:
            #     report = RootCauseReport.model_validate(
            #         structured_response
            #     )
            
            console.print()
            
            # Extract and display final result
            final_content = extract_content_from_result(result)
            display_formatted_result(final_content, streamer)
            
            break  # Success, exit retry loop
                
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                retry_count += 1
                
                # Extract wait time from error message
                import re
                wait_match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                wait_time = float(wait_match.group(1)) if wait_match else 10
                
                if retry_count < max_retries:
                    streamer.warning(f"⏳ Rate limit hit. Waiting {wait_time:.1f} seconds before retry ({retry_count}/{max_retries})...")
                    time.sleep(wait_time + 1)  # Add 1 second buffer
                    console.print()
                    streamer.info("🔄 Retrying...")
                else:
                    streamer.error(f"❌ Rate limit exceeded after {max_retries} retries.")
                    streamer.info("💡 Solutions:")
                    streamer.info("  1. Wait a minute and try again")
                    streamer.info("  2. Upgrade your Google Gemini API plan")
                    streamer.info("  3. Use a different model with higher limits")
                    break
            else:
                # Other errors
                streamer.error(f"Investigation failed: {error_msg}")
                console.print_exception()
                break


def extract_content_from_result(result):
    """Extract text content or structured object from various result formats."""
    if not result:
        return "No result returned"
    
    # Handle messages list
    if 'messages' in result and result['messages']:
        last_message = result['messages'][-1]
        
        # Handle AIMessage with content
        if hasattr(last_message, 'content'):
            content = last_message.content
            
            # Check if content is a RootCauseReport object
            from domain.rca import RootCauseReport
            if isinstance(content, RootCauseReport):
                return content
            
            # Handle list of content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                return '\n'.join(text_parts)
            
            # Handle string content
            elif isinstance(content, str):
                return content
            
            # Handle dict content
            elif isinstance(content, dict):
                # Try to parse as RootCauseReport
                try:
                    return RootCauseReport(**content)
                except:
                    if 'text' in content:
                        return content['text']
        
        # Fallback to string representation
        return str(last_message)
    
    # Fallback to string representation
    return str(result)


def display_formatted_result(content, streamer: AgentStatusStreamer):
    """Display the result in a formatted, structured way."""
    from domain.rca import RootCauseReport
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.markdown import Markdown
    import re
    
    # Handle RootCauseReport object directly
    if isinstance(content, RootCauseReport):
        display_rca_object(content, streamer)
        return
    
    # Handle string content
    if isinstance(content, str):
        # Try to parse structured RCA format
        if "**ISSUE SUMMARY**" in content or "**ROOT CAUSE**" in content:
            display_structured_rca(content, streamer)
        else:
            # Otherwise, display as formatted text
            streamer.display_result("🎯 Investigation Results", content)
    else:
        # Fallback
        streamer.display_result("🎯 Investigation Results", str(content))


def display_rca_object(report: 'RootCauseReport', streamer: AgentStatusStreamer):
    """Display a RootCauseReport Pydantic object with rich formatting."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    
    console.print()
    
    # Issue Summary
    if report.issue_summary:
        console.print(Panel(
            report.issue_summary,
            title="[bold red]📋 Issue Summary[/]",
            border_style="red",
            box=box.ROUNDED
        ))
        console.print()
    
    # Root Cause
    if report.root_cause:
        display_text = f"[bold]{report.root_cause}[/]\n\n[dim]Confidence: {report.confidence:.0%}[/]"
        console.print(Panel(
            display_text,
            title="[bold yellow]🔍 Root Cause Analysis[/]",
            border_style="yellow",
            box=box.ROUNDED
        ))
        console.print()
    
    # Expected vs Actual State
    if report.expected_state or report.actual_state:
        state_table = Table(title="Expected vs Actual State", box=box.SIMPLE, show_header=True)
        state_table.add_column("Expected State", style="green", width=40)
        state_table.add_column("Actual State", style="red", width=40)
        state_table.add_row(report.expected_state, report.actual_state)
        console.print(state_table)
        console.print()
    
    # Evidence Data (result_records)
    if report.result_records:
        streamer.display_table(report.result_records, title="📊 Evidence Data")
        console.print()
    
    # Affected Components
    if report.affected_components:
        console.print("[bold cyan]🎯 Affected Components:[/]")
        for component in report.affected_components:
            console.print(f"  • {component}")
        console.print()
    
    # Suggested Actions
    if report.suggested_actions:
        console.print("[bold green]✅ Suggested Actions:[/]")
        for i, action in enumerate(report.suggested_actions, 1):
            console.print(f"  {i}. {action}")
        console.print()
    
    # Customer Response
    if report.customer_response:
        console.print(Panel(
            report.customer_response,
            title="[bold blue]💬 Customer Response[/]",
            border_style="blue",
            box=box.ROUNDED
        ))
        console.print()
    
    # Engineering Note
    if report.engineering_note:
        console.print(Panel(
            report.engineering_note,
            title="[bold magenta]🔧 Engineering Note[/]",
            border_style="magenta",
            box=box.ROUNDED
        ))
        console.print()
    
    # Missing Information
    if report.missing_information:
        console.print("[bold yellow]⚠️  Missing Information:[/]")
        for info in report.missing_information:
            console.print(f"  • {info}")
        console.print()



def display_structured_rca(content: str, streamer: AgentStatusStreamer):
    """Parse and display structured RCA output."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.markdown import Markdown
    import re
    
    console.print()
    
    # Extract sections
    sections = {
        'issue_summary': r'\*\*ISSUE SUMMARY\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'root_cause': r'\*\*ROOT CAUSE\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'expected_state': r'\*\*EXPECTED STATE\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'actual_state': r'\*\*ACTUAL STATE\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'evidence_data': r'\*\*EVIDENCE DATA\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'affected_components': r'\*\*AFFECTED COMPONENTS\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'suggested_actions': r'\*\*SUGGESTED ACTIONS\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'customer_response': r'\*\*CUSTOMER RESPONSE\*\*\s*\n(.*?)(?=\n\*\*|$)',
        'engineering_note': r'\*\*ENGINEERING NOTE\*\*\s*\n(.*?)(?=\n\*\*|$)',
    }
    
    extracted = {}
    for key, pattern in sections.items():
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
    
    # Display Issue Summary
    if 'issue_summary' in extracted:
        console.print(Panel(
            extracted['issue_summary'],
            title="[bold red]📋 Issue Summary[/]",
            border_style="red",
            box=box.ROUNDED
        ))
        console.print()
    
    # Display Root Cause with confidence
    if 'root_cause' in extracted:
        root_cause_text = extracted['root_cause']
        # Extract confidence if present
        confidence_match = re.search(r'Confidence:\s*(\d+)%', root_cause_text)
        if confidence_match:
            confidence = confidence_match.group(1)
            root_cause_clean = re.sub(r'\s*Confidence:.*', '', root_cause_text)
            display_text = f"[bold]{root_cause_clean}[/]\n\n[dim]Confidence: {confidence}%[/]"
        else:
            display_text = root_cause_text
        
        console.print(Panel(
            display_text,
            title="[bold yellow]🔍 Root Cause Analysis[/]",
            border_style="yellow",
            box=box.ROUNDED
        ))
        console.print()
    
    # Display Expected vs Actual State
    if 'expected_state' in extracted or 'actual_state' in extracted:
        state_table = Table(title="Expected vs Actual State", box=box.SIMPLE, show_header=True)
        state_table.add_column("Expected State", style="green", width=40)
        state_table.add_column("Actual State", style="red", width=40)
        state_table.add_row(
            extracted.get('expected_state', 'N/A'),
            extracted.get('actual_state', 'N/A')
        )
        console.print(state_table)
        console.print()
    
    # Display Evidence Data Table
    if 'evidence_data' in extracted:
        evidence = extracted['evidence_data']
        console.print("[bold cyan]📊 Evidence Data[/]")
        console.print()
        # Try to parse markdown table
        lines = evidence.strip().split('\n')
        if len(lines) >= 2 and '|' in lines[0]:
            # Parse markdown table
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            
            # Header row
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]
            for header in headers:
                table.add_column(header, style="cyan")
            
            # Data rows (skip separator line)
            for line in lines[2:]:
                if '|' in line and line.strip():
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        table.add_row(*cells)
            
            console.print(table)
        else:
            console.print(evidence)
        console.print()
    
    # Display Affected Components
    if 'affected_components' in extracted:
        console.print("[bold cyan]🎯 Affected Components:[/]")
        components = extracted['affected_components'].strip().split('\n')
        for component in components:
            if component.strip():
                console.print(f"  {component.strip()}")
        console.print()
    
    # Display Suggested Actions
    if 'suggested_actions' in extracted:
        console.print("[bold green]✅ Suggested Actions:[/]")
        actions = extracted['suggested_actions'].strip().split('\n')
        for action in actions:
            if action.strip():
                console.print(f"  {action.strip()}")
        console.print()
    
    # Display Customer Response
    if 'customer_response' in extracted:
        console.print(Panel(
            extracted['customer_response'],
            title="[bold blue]💬 Customer Response[/]",
            border_style="blue",
            box=box.ROUNDED
        ))
        console.print()
    
    # Display Engineering Note
    if 'engineering_note' in extracted:
        console.print(Panel(
            extracted['engineering_note'],
            title="[bold magenta]🔧 Engineering Note[/]",
            border_style="magenta",
            box=box.ROUNDED
        ))
        console.print()



if __name__ == "__main__":
    main()
