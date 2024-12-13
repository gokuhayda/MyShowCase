
# Site-Chat-LLM

Este projeto combina um **frontend hospedado no GitHub Pages** com um **backend Flask** para interagir com um chatbot baseado em modelos avançados de linguagem.

---

## **Estrutura do Projeto**

```
site-chat-llm/
├── app/                       # Lógica do backend
│   ├── chatbot.py             # Gerenciamento do chatbot
│   ├── agent.py               # Agentes e lógica de processamento
│   ├── sessions_manager.py    # Gerenciamento de sessões
│   └── utils.py               # Funções utilitárias
├── config/                    # Configurações gerais
│   └── config.yaml
├── datasets/                  # Dados utilizados no projeto
│   ├── raw_data/              # Dados brutos
│   ├── processed_texts/       # Dados processados
├── static/                    # Arquivos estáticos (CSS, JS, imagens)
│   ├── css/                   # Estilos CSS
│   ├── js/                    # Scripts JS
│   ├── images/                # Imagens do site
├── templates/                 # Páginas HTML
│   ├── index.html             # Página inicial
│   ├── contact.html           # Página de contato
│   └── outros arquivos HTML
├── .env                       # Configurações de ambiente
├── .gitignore                 # Arquivos ignorados pelo Git
├── LICENSE                    # Licença do projeto
├── README.md                  # Documentação do projeto
```

---

## **Funcionalidades**

- **Frontend Responsivo:** Design amigável e responsivo.
- **Backend Dinâmico:** Suporte para execução local e em produção.
- **Chatbot Baseado em LLM:** Interação inteligente por meio de modelos avançados de linguagem.

---

## **Configuração e Execução**

### **1. Backend**
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Inicie o servidor Flask:
   ```bash
   python app/chatbot.py
   ```
3. (Opcional) Exponha o servidor com Ngrok:
   ```bash
   ngrok http 5000
   ```

### **2. Frontend**
1. Configure o GitHub Pages:
   - Acesse **Settings > Pages** no repositório.
   - Configure a branch e o diretório (`/` ou `/docs`).
2. Verifique que os arquivos HTML e estáticos estão corretamente apontados para o backend.

---

## **Deploy em Produção**

### **Render**
1. Configure o repositório no painel do Render.
2. Certifique-se de que os arquivos `Dockerfile` e `render.yaml` estão configurados.

### **Docker**
1. Construa e execute o container:
   ```bash
   docker-compose up --build
   ```

---

## **Contribuição**

Contribuições são bem-vindas! Para contribuir:
1. Faça um fork do repositório.
2. Crie uma branch para suas alterações:
   ```bash
   git checkout -b minha-feature
   ```
3. Realize as alterações e faça um commit.
4. Abra um pull request.

---

## **Links Importantes**

- [Perfil no LinkedIn](https://br.linkedin.com/in/éric-sena)
- [Página de Teste](https://gokuhayda.github.io/nextgen_frontend/index.html)

---

## **Licença**

Este projeto está sob a licença MIT.
