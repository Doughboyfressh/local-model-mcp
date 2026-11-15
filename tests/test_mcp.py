import asyncio
import json
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_real_stdio_protocol_and_weight_edit(workspace):
    configuration = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_model_mcp.server",
            "--root",
            str(workspace.root),
            "--print-mcp-config",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    launch = json.loads(configuration.stdout)["mcpServers"]["local-model-lab"]

    async def scenario():
        parameters = StdioServerParameters(
            command=launch["command"],
            args=launch["args"],
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                listing = await client.list_tools()
                tools = {tool.name: tool for tool in listing.tools}
                assert len(tools) == 17
                assert tools["read_tensor"].annotations.readOnlyHint
                assert tools["apply_weight_edits"].annotations.destructiveHint is False
                capabilities = await client.call_tool("capabilities", {})
                assert capabilities.structuredContent["devices"]["cpu"]
                inspect = await client.call_tool("inspect_model", {"model": "tiny", "limit": 1})
                tensor = inspect.structuredContent["tensors"][0]["name"]
                request = {"model": "tiny", "edits": [{"tensor": tensor, "operation": "zero"}]}
                preview = await client.call_tool("preview_weight_edits", {"request": request})
                assert not preview.isError
                result = await client.call_tool(
                    "apply_weight_edits",
                    {
                        "request": request,
                        "output": "via-mcp",
                        "expected_plan_token": preview.structuredContent["plan_token"],
                    },
                )
                assert not result.isError
                assert result.structuredContent["model"] == "via-mcp"
                invalid = await client.call_tool("inspect_model", {"model": "../../escape"})
                assert invalid.isError
                bad_plan = await client.call_tool(
                    "apply_weight_edits",
                    {
                        "request": request,
                        "output": "bad-plan",
                        "expected_plan_token": "wrong",
                    },
                )
                assert bad_plan.isError
                job = await client.call_tool(
                    "start_generation",
                    {
                        "request": {
                            "model": "tiny",
                            "prompt": "hello world",
                            "max_new_tokens": 2,
                            "device": "cpu",
                        }
                    },
                )
                job_id = job.structuredContent["job_id"]
                for _ in range(600):
                    response = await client.call_tool("job_status", {"job_id": job_id})
                    status = response.structuredContent
                    if status["status"] in {"completed", "failed", "interrupted"}:
                        assert status["status"] == "completed", status
                        break
                    await asyncio.sleep(0.1)
                else:
                    await client.call_tool("cancel_job", {"job_id": job_id})
                    raise AssertionError("MCP generation job timed out")

    asyncio.run(scenario())
