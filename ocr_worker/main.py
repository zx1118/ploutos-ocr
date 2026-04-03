"""
Main Entry Point
================

Entry points for OCR Worker and API.
"""

import argparse
import sys


def run_worker():
    """运行OCR消费者工作进程。
    
    此函数启动OCR引擎的消费者线程，用于从Redis流中获取任务并执行OCR识别。
    """
    from .consumer import run_consumer
    run_consumer()


def run_api():
    """运行FastAPI服务器。
    
    启动API服务，提供REST接口供外部调用OCR功能。
    """
    from .api.main import run_api as _run_api
    _run_api()


def main():
    """主入口点，处理命令行参数。
    
    解析命令行参数并根据用户输入执行相应的操作：
    - worker: 运行OCR工作进程
    - api: 运行API服务器
    - both: 同时运行工作进程和API服务器
    """
    parser = argparse.ArgumentParser(
        description="Ploutos OCR Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用的命令选项")
    
    # Worker命令解析器
    worker_parser = subparsers.add_parser(
        "worker",
        help="运行OCR工作进程 (Redis流消费者)",
    )
    worker_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过OCR引擎的预热过程",
    )
    
    # API命令解析器
    api_parser = subparsers.add_parser(
        "api",
        help="运行FastAPI服务器",
    )
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API服务器绑定的主机地址 (默认: 0.0.0.0)",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="API服务器端口 (默认: 8100)",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="启用开发模式下的自动重载功能",
    )
    
    # Both命令解析器 - 同时运行工作进程和API
    both_parser = subparsers.add_parser(
        "both",
        help="同时运行工作进程和API服务器",
    )
    
    args = parser.parse_args()
    
    if args.command == "worker":
        # 执行工作进程
        run_worker()
    elif args.command == "api":
        # 根据CLI参数覆盖设置
        from .config import settings
        settings.api.host = args.host
        settings.api.port = args.port
        run_api()
    elif args.command == "both":
        # 在单独的线程中运行两者
        import threading
        
        def run_worker_with_exception_handling():
            """工作进程包装器，包含异常处理以用于守护线程。
            
            当工作进程出现异常时，记录错误日志并确保程序稳定运行。
            """
            try:
                run_worker()
            except Exception as e:
                from loguru import logger
                logger.exception(f"工作进程线程崩溃: {e}")
        
        # 创建并启动工作进程线程
        worker_thread = threading.Thread(
            target=run_worker_with_exception_handling, 
            daemon=True,  # 设置为守护线程，主程序退出时该线程也会退出
            name="ocr-worker"  # 线程名称便于调试和监控
        )
        worker_thread.start()
        
        # 给工作进程一些时间来启动
        import time
        time.sleep(1)
        
        # 启动API服务器
        run_api()
    else:
        # 如果没有指定有效命令，则显示帮助信息并退出
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # 当作为主脚本运行时，执行主函数
    main()