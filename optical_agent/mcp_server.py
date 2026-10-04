"""Optional official MCP v2 server; imports do not start web/LLM/memory services."""
import json
import logging
import sys

from optical_agent import __version__
from optical_agent.diagnostics import redact
from optical_agent.mcp_adapter import VisionMCPAdapter


class RedactedFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


def configure_logging(log_path=None):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    for handler in handlers:
        handler.setFormatter(RedactedFormatter('%(asctime)s %(levelname)s %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def create_server(adapter=None):
    # Optional dependency stays out of every native/LangGraph/web import path.
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    service = adapter or VisionMCPAdapter()
    def response(name, arguments):
        value = service.call(name, arguments)
        return CallToolResult(content=[TextContent(type='text', text=json.dumps(value, ensure_ascii=False))],
                              structured_content=value, is_error=not value['ok'])

    class ContractServer(MCPServer):
        async def call_tool(self, name, arguments, context=None):
            fields = {'assess_image_quality': {'image_path', 'image_id'},
                      'generate_exposure_candidate': {'image_id', 'method'},
                      'detect_objects': {'image_id', 'target_classes'}}
            if name not in fields or not isinstance(arguments, dict) or set(arguments) - fields[name]:
                # Unknown tools and extra fields also get a machine-readable error.
                return response(name, arguments)
            try:
                return await super().call_tool(name, arguments, context)
            except ToolError:
                value = {'ok': False, 'tool': name, 'summary': '工具参数的字段或类型不符合要求。',
                         'error': {'category': 'parameters', 'message': '工具参数的字段或类型不符合要求。'}}
                return CallToolResult(content=[TextContent(type='text', text=json.dumps(value, ensure_ascii=False))],
                                      structured_content=value, is_error=True)

    server = ContractServer('Underwater Optical Tools', version=__version__, instructions=(
        '三个本地视觉工具；先登记图片并评估曝光，再按需校正或检测。'
        '只返回候选识别，不提供完整可靠性验收。图片通过本地路径查看。'))
    hints = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                            idempotent_hint=True, open_world_hint=False)

    @server.tool(annotations=hints)
    def assess_image_quality(image_path: str | None = None, image_id: str | None = None) -> CallToolResult:
        """测量水下曝光质量。首次传数据目录内 image_path，后续传 image_id；必须二选一。返回真实指标和热图路径。"""
        return response('assess_image_quality', dict(image_path=image_path, image_id=image_id))

    @server.tool(annotations=hints)
    def generate_exposure_candidate(image_id: str, method: str) -> CallToolResult:
        """从已评估原图生成候选；method 为 gamma_only 或 local_bounded，每种一次，重复复用。质量失败拒绝恢复。"""
        return response('generate_exposure_candidate', dict(image_id=image_id, method=method))

    @server.tool(annotations=hints)
    def detect_objects(image_id: str, target_classes: list[str] | None = None) -> CallToolResult:
        """SODD 六类候选检测；省略类别检测全部，管道对应 pipe/pipe_type2。分数不能代表可靠性，模型缺失明确报错。"""
        return response('detect_objects', dict(image_id=image_id, target_classes=target_classes))

    return server


def main():
    from optical_agent.mcp_adapter import MCP_ROOT
    configure_logging(MCP_ROOT / 'logs' / 'server.log')
    try:
        create_server().run(transport='stdio')
    except ImportError:
        logging.error('MCP 依赖不可用，请使用 D 盘 MCP 隔离环境。')
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
