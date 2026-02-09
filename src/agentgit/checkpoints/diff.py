"""Diff functionality for comparing checkpoints.

Provides utilities to compare two checkpoints and identify differences in state,
conversation history, and tool invocations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class ChangeType(Enum):
    """Type of change detected in a diff."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class StateChange:
    """Represents a change in session state."""
    path: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass
class ToolInvocationChange:
    """Represents a tool invocation in the diff."""
    index: int
    tool_name: str
    args: Dict[str, Any]
    result: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class DiffReport:
    """Complete diff report between two checkpoints."""
    checkpoint_a_id: int
    checkpoint_b_id: int
    state_changes: List[StateChange] = field(default_factory=list)
    tool_invocations: List[ToolInvocationChange] = field(default_factory=list)
    conversation_diff: Dict[str, Any] = field(default_factory=dict)
    metadata_diff: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "checkpoint_a_id": self.checkpoint_a_id,
            "checkpoint_b_id": self.checkpoint_b_id,
            "state_changes": [s.to_dict() for s in self.state_changes],
            "tool_invocations": [t.to_dict() for t in self.tool_invocations],
            "conversation_diff": self.conversation_diff,
            "metadata_diff": self.metadata_diff,
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary with dot notation paths."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _compare_dicts(old: Dict[str, Any], new: Dict[str, Any]) -> List[StateChange]:
    """Compare two dictionaries and return state changes."""
    changes = []
    
    # Flatten both dicts for comparison
    old_flat = _flatten_dict(old)
    new_flat = _flatten_dict(new)
    
    # Find all keys
    all_keys = set(old_flat.keys()) | set(new_flat.keys())
    
    for key in sorted(all_keys):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        
        if key not in old_flat:
            changes.append(StateChange(
                path=key,
                change_type=ChangeType.ADDED,
                new_value=new_val
            ))
        elif key not in new_flat:
            changes.append(StateChange(
                path=key,
                change_type=ChangeType.REMOVED,
                old_value=old_val
            ))
        elif old_val != new_val:
            changes.append(StateChange(
                path=key,
                change_type=ChangeType.MODIFIED,
                old_value=old_val,
                new_value=new_val
            ))
    
    return changes


def _compare_tool_invocations(
    old_tools: List[Dict[str, Any]],
    new_tools: List[Dict[str, Any]]
) -> List[ToolInvocationChange]:
    """Extract tool invocations that exist in new but not in old."""
    changes = []
    
    # Only show tools in B that are after A
    # Assume tools are chronologically ordered
    old_count = len(old_tools)
    new_tools_after = new_tools[old_count:]
    
    for idx, tool_inv in enumerate(new_tools_after, start=old_count):
        changes.append(ToolInvocationChange(
            index=idx,
            tool_name=tool_inv.get("tool_name", tool_inv.get("tool", "unknown")),
            args=tool_inv.get("args", {}),
            result=tool_inv.get("result"),
            success=tool_inv.get("success", True),
            error_message=tool_inv.get("error_message")
        ))
    
    return changes


def _compare_conversations(
    old_history: List[Dict[str, Any]],
    new_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare conversation histories."""
    old_count = len(old_history)
    new_count = len(new_history)
    
    return {
        "old_length": old_count,
        "new_length": new_count,
        "messages_added": new_count - old_count,
        "new_messages": new_history[old_count:] if new_count > old_count else []
    }


def _order_checkpoints(checkpoint_a, checkpoint_b):
    """Return checkpoints ordered from older to newer."""
    created_a = getattr(checkpoint_a, "created_at", None)
    created_b = getattr(checkpoint_b, "created_at", None)
    if created_a and created_b:
        if created_a <= created_b:
            return checkpoint_a, checkpoint_b
        return checkpoint_b, checkpoint_a
    
    id_a = getattr(checkpoint_a, "id", None)
    id_b = getattr(checkpoint_b, "id", None)
    if id_a is not None and id_b is not None:
        if id_a <= id_b:
            return checkpoint_a, checkpoint_b
        return checkpoint_b, checkpoint_a
    
    history_a = checkpoint_a.conversation_history or []
    history_b = checkpoint_b.conversation_history or []
    if len(history_a) != len(history_b):
        if len(history_a) <= len(history_b):
            return checkpoint_a, checkpoint_b
        return checkpoint_b, checkpoint_a
    
    tools_a = checkpoint_a.tool_invocations or []
    tools_b = checkpoint_b.tool_invocations or []
    if len(tools_a) != len(tools_b):
        if len(tools_a) <= len(tools_b):
            return checkpoint_a, checkpoint_b
        return checkpoint_b, checkpoint_a
    
    return checkpoint_a, checkpoint_b


def compare_checkpoints(checkpoint_a, checkpoint_b) -> DiffReport:
    """
    Compare two checkpoints and return a detailed diff report.
    
    Args:
        checkpoint_a: First checkpoint
        checkpoint_b: Second checkpoint
    
    Returns:
        DiffReport with all differences. Checkpoints are ordered internally
        from older to newer before comparison.
    """
    older, newer = _order_checkpoints(checkpoint_a, checkpoint_b)
    report = DiffReport(
        checkpoint_a_id=older.id,
        checkpoint_b_id=newer.id
    )
    
    # Compare session state
    old_state = older.session_state or {}
    new_state = newer.session_state or {}
    report.state_changes = _compare_dicts(old_state, new_state)
    
    # Compare tool invocations
    old_tools = older.tool_invocations or []
    new_tools = newer.tool_invocations or []
    report.tool_invocations = _compare_tool_invocations(old_tools, new_tools)
    
    # Compare conversation history
    old_conv = older.conversation_history or []
    new_conv = newer.conversation_history or []
    report.conversation_diff = _compare_conversations(old_conv, new_conv)
    
    # Compare metadata
    old_meta = older.metadata or {}
    new_meta = newer.metadata or {}
    report.metadata_diff = {
        "old_metadata": old_meta,
        "new_metadata": new_meta,
    }
    
    return report


def format_diff_report(report: DiffReport, json_output: bool = False) -> str:
    """
    Format a diff report for display.
    
    Args:
        report: The DiffReport to format
        json_output: If True, return JSON; otherwise human-readable text
    
    Returns:
        Formatted report as string
    """
    if json_output:
        return report.to_json()
    
    lines = []
    lines.append(f"Checkpoint Diff: {report.checkpoint_a_id} → {report.checkpoint_b_id}")
    lines.append("=" * 70)
    
    # State changes
    if report.state_changes:
        lines.append("\n📊 STATE CHANGES:")
        lines.append("-" * 70)
        
        added = [c for c in report.state_changes if c.change_type == ChangeType.ADDED]
        removed = [c for c in report.state_changes if c.change_type == ChangeType.REMOVED]
        modified = [c for c in report.state_changes if c.change_type == ChangeType.MODIFIED]
        
        if added:
            lines.append("\n  ➕ ADDED:")
            for change in added:
                lines.append(f"    {change.path}: {change.new_value}")
        
        if removed:
            lines.append("\n  ➖ REMOVED:")
            for change in removed:
                lines.append(f"    {change.path}: {change.old_value}")
        
        if modified:
            lines.append("\n  🔄 MODIFIED:")
            for change in modified:
                lines.append(f"    {change.path}:")
                lines.append(f"      - {change.old_value}")
                lines.append(f"      + {change.new_value}")
    else:
        lines.append("\n📊 STATE CHANGES: None")
    
    # Tool invocations
    if report.tool_invocations:
        lines.append("\n\n🔧 TOOL INVOCATIONS (after checkpoint A):")
        lines.append("-" * 70)
        for tool in report.tool_invocations:
            status = "✓" if tool.success else "✗"
            lines.append(f"\n  {status} [{tool.index}] {tool.tool_name}")
            if tool.args:
                lines.append(f"      Args: {tool.args}")
            if tool.result:
                lines.append(f"      Result: {tool.result}")
            if tool.error_message:
                lines.append(f"      Error: {tool.error_message}")
    else:
        lines.append("\n\n🔧 TOOL INVOCATIONS: None")
    
    # Conversation changes
    conv_diff = report.conversation_diff
    lines.append(f"\n\n💬 CONVERSATION:")
    lines.append("-" * 70)
    lines.append(f"  Messages in A: {conv_diff.get('old_length', 0)}")
    lines.append(f"  Messages in B: {conv_diff.get('new_length', 0)}")
    lines.append(f"  Messages added: {conv_diff.get('messages_added', 0)}")
    
    if conv_diff.get('new_messages'):
        lines.append(f"\n  New messages:")
        for msg in conv_diff['new_messages']:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')[:100]
            lines.append(f"    [{role}] {content}...")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


class CheckpointDiffer:
    """Compatibility wrapper for checkpoint diffing."""
    
    @staticmethod
    def diff(checkpoint_a, checkpoint_b) -> DiffReport:
        """Compare two checkpoints and return a diff report."""
        return compare_checkpoints(checkpoint_a, checkpoint_b)


class DiffRenderer:
    """Compatibility wrapper for diff rendering."""
    
    @staticmethod
    def render_text(report: DiffReport, use_color: bool = False) -> str:
        """Render a diff report as text, optionally with ANSI colors."""
        output = format_diff_report(report, json_output=False)
        if not use_color:
            return output
        
        # Simple ANSI styling for headers and section labels.
        bold = "\033[1m"
        cyan = "\033[96m"
        dim = "\033[2m"
        reset = "\033[0m"
        
        lines = []
        for line in output.splitlines():
            if line.startswith("Checkpoint Diff:"):
                lines.append(f"{bold}{line}{reset}")
            elif line.startswith("=" * 70):
                lines.append(f"{dim}{line}{reset}")
            elif "STATE CHANGES" in line or "TOOL INVOCATIONS" in line or "CONVERSATION" in line:
                lines.append(f"{cyan}{bold}{line}{reset}")
            else:
                lines.append(line)
        return "\n".join(lines)
