import typer
from rich import print
import sys
import signal
import json
import socket
import os
import threading
from RAG_project.agent import RAGAgent
from FT_project.agent import FTAgent

app = typer.Typer(help="Intelligent Assistant CLI")

# Unix 域套接字设置
RAG_SOCKET = "/tmp/rag_socket"  # RAG 服务监听的套接字路径
FT_SOCKET = "/tmp/ft_socket"    # FT 服务监听的套接字路径

def socket_server(socket_path, message_handler):
    """启动 Unix 域套接字服务器来接收消息"""
    # 如果套接字文件已存在，则删除它
    if os.path.exists(socket_path):
        os.unlink(socket_path)
        
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(socket_path)
        server_socket.listen(1)
        
        while True:
            try:
                conn, _ = server_socket.accept()
                with conn:
                    data = conn.recv(4096)
                    if data:
                        message = json.loads(data.decode())
                        message_handler(message)
            except Exception as e:
                print(f"[red]Socket error: {str(e)}[/red]")
                # 如果出现异常，短暂休眠避免CPU占用过高
                import time
                time.sleep(0.1)

def send_socket_message(socket_path, message):
    """发送消息到指定路径的 Unix 域套接字服务器"""
    try:
        if not os.path.exists(socket_path):
            print(f"[yellow]Socket {socket_path} doesn't exist. Recipient process may not be running.[/yellow]")
            return False
            
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(socket_path)
            client_socket.sendall(json.dumps(message).encode())
        return True
    except Exception as e:
        print(f"[red]Error sending message: {str(e)}[/red]")
        return False

def rag_process_main():
    """RAG 进程的主函数"""
    # 初始化 RAG Agent
    rag_agent = RAGAgent()
    
    # 定义消息处理函数
    def handle_message(message):
        if message.get("type") == "update_settings":
            params = message.get("params")
            rag_agent.update_settings(params)
            print("[green] \n RAG settings updated with fine-tuned model[/green]")
    
    # 在后台线程中启动套接字服务器
    server_thread = threading.Thread(target=socket_server, args=(RAG_SOCKET, handle_message))
    server_thread.daemon = True  # 设为守护线程，主进程退出时自动结束
    server_thread.start()
    
    print("[green]RAG process started[/green]")
    print(f"[blue]Listening for messages on {RAG_SOCKET}[/blue]")
    
    # 处理信号以便优雅退出
    def signal_handler(sig, frame):
        print("[yellow]RAG process received exit signal[/yellow]")
        # 清理套接字文件
        if os.path.exists(RAG_SOCKET):
            os.unlink(RAG_SOCKET)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主循环
    while True:
        try:
            # 处理用户问题
            question = typer.prompt("\nEnter a question for RAG (type 'exit' to stop)")
            
            if question.lower() == "exit":
                # 清理套接字文件
                if os.path.exists(RAG_SOCKET):
                    os.unlink(RAG_SOCKET)
                break
                
            # 获取回答
            answer = rag_agent.respond(question)
            print(f"[bold green]RAG Answer:[/bold green]\n{answer}")
            
            # 收集反馈
            print("\n[cyan]Please provide feedback to help us improve:[/cyan]")
            rating = typer.prompt("Rating (1-5, where 1-3 means incorrect, 5 is the best)", type=int)
            
            if rating < 4:
                correct_answer = typer.prompt("Please provide the correct answer (press Enter to skip)", default="")
                if correct_answer.strip():
                    rag_agent.save_feedback(
                        question=question,
                        model_answer=answer,
                        rating=rating,
                        correct_answer=correct_answer
                    )
                    
                    # 告诉 FT 进程有新的反馈数据
                    feedback_message = {
                        "type": "new_feedback",
                        "path": "./feedback_log/rag_feedback.log"  # 假设反馈保存在这个路径
                    }
                    
                    if send_socket_message(FT_SOCKET, feedback_message):
                        print("[green]✓ Thank you for your feedback! FT process notified.[/green]")
                    else:
                        print("[yellow]✓ Feedback saved, but couldn't notify FT process.[/yellow]")
                else:
                    print("[yellow]No correct answer provided. Feedback not saved.[/yellow]")
            else:
                print("[green]✓ Thank you for your rating![/green]")
        
        except Exception as e:
            print(f"[red]RAG Error:[/red] {str(e)}")

def ft_process_main():
    """FT 进程的主函数"""
    # 初始化 FT Agent
    ft_agent = FTAgent()
    
    # 定义消息处理函数
    def handle_message(message):
        if message.get("type") == "new_feedback":
            feedback_path = message.get("path")
            print(f"\n[blue]New feedback available at {feedback_path}[/blue]")
            # 这里你可以选择是否自动开始微调
    
    # 在后台线程中启动套接字服务器
    server_thread = threading.Thread(target=socket_server, args=(FT_SOCKET, handle_message))
    server_thread.daemon = True  # 设为守护线程，主进程退出时自动结束
    server_thread.start()
    
    print("[green]FT process started[/green]")
    print(f"[blue]Listening for messages on {FT_SOCKET}[/blue]")
    
    # 处理信号以便优雅退出
    def signal_handler(sig, frame):
        print("[yellow]FT process received exit signal[/yellow]")
        # 清理套接字文件
        if os.path.exists(FT_SOCKET):
            os.unlink(FT_SOCKET)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主循环
    while True:
        try:
            # 用户命令处理
            command = typer.prompt("\nEnter command ('train' to start fine-tuning, 'exit' to stop)")
            
            if command.lower() == "exit":
                # 清理套接字文件
                if os.path.exists(FT_SOCKET):
                    os.unlink(FT_SOCKET)
                break
                
            elif command.lower() == "train":
                # 获取训练数据路径
                data_path = typer.prompt("Enter the path for fine-tuning data")
                
                print("[blue]Starting fine-tuning...[/blue]")
                
                # 执行微调
                result = ft_agent.train(data_path)
                print("[green]Training completed![/green]")
                
                fine_tuned_model_path = result.get("best_model_checkpoint")
                
                if fine_tuned_model_path:
                    # 更新自身设置
                    update_params = {"ADAPTER_DIR": fine_tuned_model_path}
                    ft_agent.update_settings(update_params)
                    
                    # 通知 RAG 进程更新设置
                    update_message = {
                        "type": "update_settings",
                        "params": update_params
                    }
                    
                    if send_socket_message(RAG_SOCKET, update_message):
                        print(f"[green]Fine-tuned model saved and RAG process notified.[/green]")
                    else:
                        print(f"[yellow]Fine-tuned model saved at: {fine_tuned_model_path}, but couldn't notify RAG process.[/yellow]")
                else:
                    print("[red]Error: Fine-tuned model path not found in result[/red]")
            
            else:
                print(f"[yellow]Unknown command: {command}[/yellow]")
        
        except Exception as e:
            print(f"[red]FT Error:[/red] {str(e)}")

@app.command()
def run_rag():
    """启动 RAG 进程"""
    rag_process_main()

@app.command()
def run_ft():
    """启动 FT 进程"""
    ft_process_main()


if __name__ == "__main__":
    app()