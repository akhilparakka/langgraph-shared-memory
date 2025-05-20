def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "todo", "Profile")
    """
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append(
                    {
                        "type": "update",
                        "doc_id": call["args"]["json_doc_id"],
                        "planned_edits": call["args"]["planned_edits"],
                        "value": call["args"]["patches"][0]["value"],
                    }
                )
            elif call["name"] == schema_name:
                changes.append({"type": "new", "value": call["args"]})

    result_parts = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n" f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)
