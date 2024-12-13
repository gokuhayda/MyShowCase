
# Site-Chat-LLM

Este é um projeto híbrido que combina um frontend hospedado no **GitHub Pages** com um backend em Flask, que pode ser executado localmente usando **Ngrok** ou em produção no **Render**. A aplicação permite a interação com um chatbot baseado em modelos de linguagem.

---

## **1. Estrutura do Projeto**

```
site-chat-llm/
├── backend/                       # Código do backend (Flask)
│   ├── app/                       # Lógica do backend
│   ├── requirements.txt           # Dependências do backend
│   ├── Dockerfile                 # Configuração Docker
│   ├── docker-compose.yml         # Configuração Docker Compose
│   ├── entrypoint.sh              # Script de inicialização
│   ├── render.yaml                # Configuração do Render
├── frontend/                      # Código do frontend
│   ├── index.html                 # Página inicial
│   ├── cases.html                 # Página de cases
│   ├── contact.html               # Página de contato
│   ├── styles.css                 # Estilos CSS
│   ├── chatbot.js                 # Integração com o backend
│   ├── images/                    # Imagens do site
│       ├── logo.webp
├── .gitignore                     # Arquivos ignorados pelo Git
├── README.md                      # Documentação do projeto
```

---

## **2. Funcionalidades**

- **Frontend Responsivo:** Hospedado no **GitHub Pages**, acessível em qualquer dispositivo.
- **Backend Dinâmico:** Gerenciado via **Render** ou **Ngrok**, permitindo fácil integração e testes.
- **Chatbot Baseado em LLM:** Utiliza modelos avançados de linguagem para interações inteligentes.

---

## **3. Instalação**

### **Frontend**

1. Navegue até a pasta `frontend/`.
2. Suba os arquivos para o GitHub Pages:
   - Vá em **Settings > Pages** no repositório do GitHub.
   - Selecione a branch correta e a pasta `/frontend`.

### **Backend**

1. Navegue até a pasta `backend/`.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicie o servidor localmente:
   ```bash
   python app/chatbot.py
   ```
4. (Opcional) Use **Ngrok** para expor o servidor:
   ```bash
   ngrok http 5000
   ```

### **Docker**

Para rodar o backend em um ambiente Dockerizado:
1. Construa e inicie o container:
   ```bash
   docker-compose up --build
   ```

---

## **4. Deploy no Render**

1. Configure o repositório do backend no **Render**.
2. Certifique-se de que o arquivo `render.yaml` e o `Dockerfile` estão na pasta `backend/`.
3. Adicione variáveis de ambiente no painel do Render, se necessário.

---

## **5. Configuração do Frontend**

O arquivo `chatbot.js` no frontend deve apontar para o backend correto. Altere a URL do backend:

```javascript
const LOCAL_BACKEND_URL = "http://localhost:5000";
const RENDER_BACKEND_URL = "https://seu-backend-render.onrender.com";
const NGROK_BACKEND_URL = "https://seu-backend-ngrok.ngrok.io";

const BACKEND_URL = RENDER_BACKEND_URL; // Atualize conforme o ambiente
```

---

## **6. Tecnologias Utilizadas**

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask, Python
- **Integração:** Ngrok, Render
- **Deploy:** GitHub Pages, Docker

---

## **7. Contribuição**

Contribuições são bem-vindas! Para contribuir:

1. Crie um fork do repositório.
2. Crie uma branch para sua feature:
   ```bash
   git checkout -b minha-feature
   ```
3. Faça commit das alterações.
4. Envie um pull request.

---

## **8. Licença**

Este projeto está sob a licença MIT.
