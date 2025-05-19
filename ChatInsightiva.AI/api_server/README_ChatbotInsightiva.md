# 📦 Chatbot Insightiva - Integração com n8n

Este pacote contém tudo o que você precisa para rodar seu próprio nó personalizado do **Chatbot Insightiva** dentro do ambiente do n8n.

---

## ✅ Conteúdo

- `ChatbotInsightiva.node.ts` - Lógica do nó
- `ChatbotInsightiva.credentials.ts` - Autenticação via API Key
- `ChatbotInsightiva.png` - Ícone personalizado
- `Chatbot_Insightiva_Fluxo_Exemplo.json` - Fluxo n8n de exemplo

---

## 🚀 Como instalar

### 1. Clone o repositório oficial do n8n
```bash
git clone https://github.com/n8n-io/n8n.git
cd n8n
```

### 2. Instale o `pnpm` (caso ainda não tenha)
```bash
npm install -g corepack
corepack enable
corepack prepare pnpm@latest --activate
```

### 3. Copie os arquivos para as pastas corretas

#### ➤ Nó
Copie `ChatbotInsightiva.node.ts` e `ChatbotInsightiva.png` para:
```
packages/nodes-base/nodes/ChatbotInsightiva/
```

#### ➤ Credencial
Copie `ChatbotInsightiva.credentials.ts` para:
```
packages/nodes-base/credentials/
```

### 4. Registre os exports

#### Em `packages/nodes-base/nodes/index.ts`:
```ts
export * from './ChatbotInsightiva/ChatbotInsightiva.node';
```

#### Em `packages/nodes-base/credentials/index.ts`:
```ts
export * from './ChatbotInsightiva.credentials';
```

### 5. Instale as dependências e rode
```bash
pnpm install
pnpm run dev
```

---

## 🧪 Teste rápido

1. Acesse `http://localhost:5678`
2. Importe o fluxo `Chatbot_Insightiva_Fluxo_Exemplo.json`
3. Configure sua API Key no nó
4. Execute o fluxo

---

Feito com 💙 por [Você]
