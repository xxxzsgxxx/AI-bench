import os
import time
import json
import readline
import requests
from argparse import ArgumentParser

class SessionManager:
    """多会话管理系统"""
    def __init__(self):
        self.sessions = {}
        self.current_id = 0

    def create_session(self):
        self.current_id += 1
        self.sessions[self.current_id] = {
            'messages': [],
            'created': time.time(),
            'token_usage': 0,
            'metrics': {}
        }
        return self.current_id

class ChatClient:
    def __init__(self, api_url, api_key, model):
        # API路径标准化处理
        self.api_url = self._normalize_api_url(api_url)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.sessions = SessionManager()
        self.current_session = self.sessions.create_session()
        self.history_file = os.path.expanduser('~/.chat_client_history')
        self._init_readline()

    def _normalize_api_url(self, url):
        """确保API路径符合OpenAI规范"""
        url = url.rstrip('/')
        if not url.endswith('/v1'):
            url += '/v1'
        return url

    def _init_readline(self):
        """初始化命令行历史功能"""
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            open(self.history_file, 'a').close()
        readline.set_history_length(100)

    def _completer(self, text, state):
        """命令自动补全逻辑"""
        commands = ['/new', '/list', '/switch', '/exit']
        matches = [c for c in commands if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    def _handle_command(self, input_cmd):
        """处理系统命令"""
        cmd = input_cmd.strip().lower()
        if cmd == '/new':
            new_id = self.sessions.create_session()
            print(f"\033[33m[新会话 {new_id} 已创建]\033[0m")
            return True
        elif cmd.startswith('/switch'):
            try:
                session_id = int(cmd.split()[1])
                if session_id in self.sessions.sessions:
                    self.current_session = session_id
                    print(f"\033[33m[已切换至会话 {session_id}]\033[0m")
                else:
                    print("\033[31m错误会话ID\033[0m")
            except:
                print("\033[31m用法: /switch <会话ID>\033[0m")
            return True
        elif cmd == '/list':
            print("\n\033[33m[活跃会话列表]")
            for sid, sess in self.sessions.sessions.items():
                stats = f"消息数: {len(sess['messages'])//2} | Tokens: {sess['token_usage']}"
                print(f"{sid}.\t{stats}")
            return True
        return False

    def _show_metrics(self, metrics):
        """显示性能统计信息"""
        if not metrics['first_token']:
            print("\033[31m未收到有效响应\033[0m")
            return

        total_time = time.time() - metrics['start_time']
        generation_time = metrics['last_token'] - metrics['first_token']

        print(f"\n\033[33m[性能统计]")
        print(f"首Token延迟: {metrics['first_token'] - metrics['start_time']:.2f}s")
        print(f"总生成时间: {generation_time:.2f}s")
        print(f"总Token数量: {metrics['total_tokens']}")
        print(f"生成速度: {metrics['total_tokens']/generation_time:.1f} tokens/s")
        print(f"端到端速度: {metrics['total_tokens']/total_time:.1f} tokens/s\033[0m")

    def stream_chat(self, prompt):
        """流式对话核心逻辑"""
        session = self.sessions.sessions[self.current_session]
        session['messages'].append({"role": "user", "content": prompt})

        metrics = {
            'start_time': time.time(),
            'first_token': None,
            'last_token': None,
            'total_tokens': 0
        }

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": session['messages'],
                    "stream": True
                },
                stream=True,
                timeout=15
            )
            response.raise_for_status()

            print("\033[32mAssistant:\033[0m ", end="", flush=True)
            full_response = []

            for chunk in response.iter_lines():
                if chunk:
                    data = chunk.decode('utf-8').lstrip('data: ').strip()
                    if not data or data == "[DONE]": continue

                    try:
                        json_data = json.loads(data)
                        content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    except json.JSONDecodeError:
                        continue

                    if content:
                        # 更新性能指标
                        metrics['total_tokens'] += len(content)
                        metrics['last_token'] = time.time()
                        if not metrics['first_token']:
                            metrics['first_token'] = metrics['last_token']

                        # 流式输出
                        print(content, end="", flush=True)
                        full_response.append(content)

            # 保存会话数据
            session['messages'].append({"role": "assistant", "content": "".join(full_response)})
            session['token_usage'] += metrics['total_tokens']
            session['metrics'] = metrics.copy()

            # 显示统计信息
            self._show_metrics(metrics)

        except requests.exceptions.HTTPError as e:
            error_msg = json.loads(e.response.text).get('error', {}).get('message', str(e))
            print(f"\n\033[31mAPI错误: {error_msg}\033[0m")
        except Exception as e:
            print(f"\n\033[31m错误: {str(e)}\033[0m")

if __name__ == "__main__":
    parser = ArgumentParser(description="大模型对话客户端")
    parser.add_argument("--api-url", required=True, help="API服务地址")
    parser.add_argument("--api-key", required=True, help="API认证密钥")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="模型名称")
    args = parser.parse_args()

    client = ChatClient(args.api_url, args.api_key, args.model)

    try:
        while True:
            try:
                user_input = input("\033[34mYou:\033[0m ")
            except KeyboardInterrupt:
                print("\n\033[33m对话已终止\033[0m")
                break

            if user_input.lower() in ('exit', 'quit'):
                break

            if client._handle_command(user_input):
                continue

            client.stream_chat(user_input)
            readline.append_history_file(1, client.history_file)

    finally:
        print("\033[33m[对话历史已保存]\033[0m")
