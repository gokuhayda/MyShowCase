import threading
import time
import ipywidgets as widgets
from IPython.display import display, clear_output
from crewai.tools.base_tool import BaseTool
from pydantic import Field

# ==============================================================================
# Classe de Ferramenta para Interações Humanas via Widget
# ==============================================================================

class HumanInputTool(BaseTool):
    """
    Ferramenta que exibe um widget no Jupyter para solicitar input humano,
    caso o agente não tenha informações suficientes para tomar uma decisão.
    """
    name: str = "Human Interaction"
    description: str = "Solicita entrada humana quando o agente não consegue tomar uma decisão."

    prompt: str = Field(default=None, description="Pergunta enviada ao humano.")
    
    def _run(self, prompt: str) -> str:
        self.prompt = prompt

        # Container para exibir os widgets
        container = widgets.VBox()
        output = widgets.Output()

        # Exibir o prompt
        prompt_label = widgets.HTML(value=f"<b>Pergunta do agente:</b> {self.prompt}")

        # Entrada de texto para a resposta
        response_widget = widgets.Text(
            placeholder="Digite sua resposta aqui",
            layout=widgets.Layout(width='80%')
        )

        # Botão para enviar a resposta
        submit_button = widgets.Button(description="Enviar resposta", button_style='success')

        # Label para exibir o tempo restante
        timer_label = widgets.Label(value="Tempo restante: 60 segundos")

        # Variável para armazenar a resposta final
        response = ""

        # Evento para sinalizar o envio da resposta
        response_event = threading.Event()

        def on_submit_click(b):
            """
            Callback para capturar a resposta do usuário quando o botão for clicado.
            """
            nonlocal response
            response = response_widget.value.strip()
            with output:
                clear_output()
                if not response:
                    print("Nenhuma resposta fornecida. Continuando sem resposta...")
                else:
                    print("Resposta enviada com sucesso!")
                    print(f"Resposta: {response}")
            response_event.set()
            container.close()

        submit_button.on_click(on_submit_click)

        def monitor_input():
            """
            Monitora o tempo e, após 60s (ou outro valor), continua automaticamente
            mesmo sem a resposta do usuário.
            """
            time_remaining = 1
            while time_remaining > 0:
                time.sleep(1)
                time_remaining -= 1
                timer_label.value = f"Tempo restante: {time_remaining} segundos"
                if response_event.is_set():
                    return  # Interrompe se a resposta já foi enviada
                # Se quiser detectar se o usuário começou a digitar:
                # if response_widget.value.strip():
                #     timer_label.value = "Resposta detectada, mas não enviada ainda."
            if not response_event.is_set():
                with output:
                    clear_output()
                    print("Aviso: Tempo esgotado. Continuando sem resposta...")
                container.close()

        # Iniciar o monitoramento em uma thread separada
        timer_thread = threading.Thread(target=monitor_input, daemon=True)
        timer_thread.start()

        # Adicionar widgets ao container
        container.children = [
            prompt_label, response_widget, submit_button, timer_label, output
        ]

        # Exibir os widgets
        display(container)

        # Bloqueia até o envio da resposta ou encerramento do timer
        response_event.wait(timeout=70)  # margenzinha de 10s
        return response

    async def _arun(self, prompt: str) -> str:
        """
        Método assíncrono não implementado, pois estamos usando widgets sincronos de Jupyter.
        """
        raise NotImplementedError("Execução assíncrona não é suportada.")