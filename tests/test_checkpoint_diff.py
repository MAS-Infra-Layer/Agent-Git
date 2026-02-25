"""Tests for checkpoint diff functionality."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import json
from datetime import datetime

from agentgit.checkpoints.checkpoint import Checkpoint
from agentgit.checkpoints.diff import (
    compare_checkpoints,
    format_diff_report,
    StateChange,
    ChangeType,
    ToolInvocationChange,
    DiffReport,
    _flatten_dict,
    _compare_dicts,
    _compare_tool_invocations,
)


class TestFlattenDict:
    """Test dictionary flattening."""
    
    def test_flatten_simple_dict(self):
        """Test flattening a simple dict."""
        d = {"a": 1, "b": 2}
        result = _flatten_dict(d)
        assert result == {"a": 1, "b": 2}
    
    def test_flatten_nested_dict(self):
        """Test flattening a nested dict."""
        d = {"a": {"b": {"c": 1}}, "x": 2}
        result = _flatten_dict(d)
        assert result == {"a.b.c": 1, "x": 2}
    
    def test_flatten_mixed_dict(self):
        """Test flattening mixed nested and flat keys."""
        d = {
            "user": {"name": "Alice", "profile": {"age": 30}},
            "count": 5
        }
        result = _flatten_dict(d)
        assert "user.name" in result
        assert "user.profile.age" in result
        assert "count" in result


class TestCompareDicts:
    """Test dict comparison."""
    
    def test_no_changes(self):
        """Test comparing identical dicts."""
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 2}
        changes = _compare_dicts(old, new)
        assert len(changes) == 0
    
    def test_added_key(self):
        """Test added keys."""
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        changes = _compare_dicts(old, new)
        
        assert len(changes) == 1
        assert changes[0].path == "b"
        assert changes[0].change_type == ChangeType.ADDED
        assert changes[0].new_value == 2
    
    def test_removed_key(self):
        """Test removed keys."""
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        changes = _compare_dicts(old, new)
        
        assert len(changes) == 1
        assert changes[0].path == "b"
        assert changes[0].change_type == ChangeType.REMOVED
        assert changes[0].old_value == 2
    
    def test_modified_value(self):
        """Test modified values."""
        old = {"a": 1, "b": "old"}
        new = {"a": 1, "b": "new"}
        changes = _compare_dicts(old, new)
        
        assert len(changes) == 1
        assert changes[0].path == "b"
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[0].old_value == "old"
        assert changes[0].new_value == "new"
    
    def test_nested_changes(self):
        """Test changes in nested dicts."""
        old = {"user": {"name": "Alice", "age": 30}}
        new = {"user": {"name": "Alice", "age": 31, "city": "NYC"}}
        changes = _compare_dicts(old, new)
        
        assert len(changes) == 2
        paths = {c.path for c in changes}
        assert "user.age" in paths
        assert "user.city" in paths


class TestCompareToolInvocations:
    """Test tool invocation comparison."""
    
    def test_no_new_tools(self):
        """Test when no new tools are invoked."""
        old = [{"tool_name": "tool1", "args": {}}]
        new = [{"tool_name": "tool1", "args": {}}]
        changes = _compare_tool_invocations(old, new)
        assert len(changes) == 0
    
    def test_new_tools_added(self):
        """Test when new tools are added."""
        old = [{"tool_name": "tool1", "args": {"x": 1}}]
        new = [
            {"tool_name": "tool1", "args": {"x": 1}},
            {"tool_name": "tool2", "args": {"y": 2}, "success": True},
            {"tool_name": "tool3", "args": {}, "success": False, "error_message": "Failed"}
        ]
        changes = _compare_tool_invocations(old, new)
        
        assert len(changes) == 2
        assert changes[0].tool_name == "tool2"
        assert changes[0].index == 1
        assert changes[0].success is True
        assert changes[1].tool_name == "tool3"
        assert changes[1].success is False
        assert changes[1].error_message == "Failed"


class TestDiffReport:
    """Test DiffReport class."""
    
    def test_report_creation(self):
        """Test creating a diff report."""
        report = DiffReport(checkpoint_a_id=1, checkpoint_b_id=2)
        assert report.checkpoint_a_id == 1
        assert report.checkpoint_b_id == 2
        assert len(report.state_changes) == 0
        assert len(report.tool_invocations) == 0
    
    def test_report_to_dict(self):
        """Test converting report to dict."""
        report = DiffReport(
            checkpoint_a_id=1,
            checkpoint_b_id=2,
            state_changes=[StateChange("key", ChangeType.ADDED, new_value=1)]
        )
        d = report.to_dict()
        assert d["checkpoint_a_id"] == 1
        assert d["checkpoint_b_id"] == 2
        assert len(d["state_changes"]) == 1
    
    def test_report_to_json(self):
        """Test converting report to JSON."""
        report = DiffReport(checkpoint_a_id=1, checkpoint_b_id=2)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["checkpoint_a_id"] == 1
        assert parsed["checkpoint_b_id"] == 2


class TestCompareCheckpoints:
    """Integration tests for checkpoint comparison."""
    
    def test_compare_identical_checkpoints(self):
        """Test comparing identical checkpoints."""
        cp = Checkpoint(
            id=1,
            internal_session_id=1,
            session_state={"counter": 5},
            tool_invocations=[{"tool_name": "tool1", "args": {}}]
        )
        
        report = compare_checkpoints(cp, cp)
        assert len(report.state_changes) == 0
        assert len(report.tool_invocations) == 0
    
    def test_compare_with_state_changes(self):
        """Test comparing checkpoints with state changes."""
        cp_a = Checkpoint(
            id=1,
            internal_session_id=1,
            session_state={"counter": 5, "name": "Alice"},
            tool_invocations=[]
        )
        
        cp_b = Checkpoint(
            id=2,
            internal_session_id=1,
            session_state={"counter": 10, "name": "Alice", "age": 30},
            tool_invocations=[]
        )
        
        report = compare_checkpoints(cp_a, cp_b)
        
        assert len(report.state_changes) == 2
        paths = {c.path for c in report.state_changes}
        assert "counter" in paths
        assert "age" in paths
    
    def test_compare_with_tool_invocations(self):
        """Test comparing checkpoints with tool invocations."""
        cp_a = Checkpoint(
            id=1,
            internal_session_id=1,
            session_state={},
            tool_invocations=[
                {"tool_name": "read_file", "args": {"path": "/file1"}, "success": True}
            ]
        )
        
        cp_b = Checkpoint(
            id=2,
            internal_session_id=1,
            session_state={},
            tool_invocations=[
                {"tool_name": "read_file", "args": {"path": "/file1"}, "success": True},
                {"tool_name": "write_file", "args": {"path": "/file2"}, "success": True},
                {"tool_name": "delete_file", "args": {"path": "/file3"}, "success": False, "error_message": "Permission denied"}
            ]
        )
        
        report = compare_checkpoints(cp_a, cp_b)
        
        assert len(report.tool_invocations) == 2
        assert report.tool_invocations[0].tool_name == "write_file"
        assert report.tool_invocations[1].tool_name == "delete_file"
        assert report.tool_invocations[1].success is False
    
    def test_compare_with_conversation_changes(self):
        """Test comparing conversation histories."""
        cp_a = Checkpoint(
            id=1,
            internal_session_id=1,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        )
        
        cp_b = Checkpoint(
            id=2,
            internal_session_id=1,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well"}
            ]
        )
        
        report = compare_checkpoints(cp_a, cp_b)
        
        assert report.conversation_diff["old_length"] == 2
        assert report.conversation_diff["new_length"] == 4
        assert report.conversation_diff["messages_added"] == 2
        assert len(report.conversation_diff["new_messages"]) == 2

    def test_compare_orders_by_created_at(self):
        """Test that checkpoints are ordered by created_at when comparing."""
        older_time = datetime(2024, 1, 1, 12, 0, 0)
        newer_time = datetime(2024, 1, 2, 12, 0, 0)
        
        cp_old = Checkpoint(
            id=10,
            internal_session_id=1,
            created_at=older_time,
            conversation_history=[
                {"role": "user", "content": "Hello"}
            ]
        )
        
        cp_new = Checkpoint(
            id=1,
            internal_session_id=1,
            created_at=newer_time,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        )
        
        # Pass them in reverse order; report should still be old -> new
        report = compare_checkpoints(cp_new, cp_old)
        
        assert report.checkpoint_a_id == 10
        assert report.checkpoint_b_id == 1
        assert report.conversation_diff["messages_added"] == 1


class TestFormatDiffReport:
    """Test formatting diff reports."""
    
    def test_format_text_output(self):
        """Test text formatting."""
        cp_a = Checkpoint(id=1, session_state={"a": 1})
        cp_b = Checkpoint(id=2, session_state={"a": 2, "b": 3})
        
        report = compare_checkpoints(cp_a, cp_b)
        output = format_diff_report(report, json_output=False)
        
        assert "Checkpoint Diff: 1 → 2" in output
        assert "STATE CHANGES" in output
        assert "TOOL INVOCATIONS" in output
        assert "CONVERSATION" in output
    
    def test_format_json_output(self):
        """Test JSON formatting."""
        cp_a = Checkpoint(id=1, session_state={})
        cp_b = Checkpoint(id=2, session_state={"key": "value"})
        
        report = compare_checkpoints(cp_a, cp_b)
        output = format_diff_report(report, json_output=True)
        
        parsed = json.loads(output)
        assert parsed["checkpoint_a_id"] == 1
        assert parsed["checkpoint_b_id"] == 2
        assert len(parsed["state_changes"]) == 1
    
    def test_format_with_tool_failures(self):
        """Test formatting with failed tools."""
        report = DiffReport(
            checkpoint_a_id=1,
            checkpoint_b_id=2,
            tool_invocations=[
                ToolInvocationChange(
                    index=0,
                    tool_name="network_call",
                    args={"url": "http://example.com"},
                    success=False,
                    error_message="Connection timeout"
                )
            ]
        )
        
        output = format_diff_report(report, json_output=False)
        assert "network_call" in output
        assert "Connection timeout" in output
        assert "✗" in output
