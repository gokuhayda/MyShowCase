#!/bin/bash

# Nome do diretório do projeto
PROJETO="n8n-custom-chatbot"
NODE_DIR="ChatbotInsightiva"

echo "🚀 Instalando corepack e pnpm..."
npm install -g corepack
corepack enable
corepack prepare pnpm@latest --activate

echo "📦 Clonando repositório oficial do n8n..."
git clone https://github.com/n8n-io/n8n.git "$PROJETO"
cd "$PROJETO"

echo "📁 Criando diretório para o novo node..."
mkdir -p packages/nodes-base/nodes/$NODE_DIR

echo "📄 Adicionando o ChatbotInsightiva.node.ts..."
cat << 'EOF' > packages/nodes-base/nodes/$NODE_DIR/${NODE_DIR}.node.ts
import { INodeProperties, INodeType, INodeTypeDescription, IExecuteFunctions } from 'n8n-workflow';
import { OptionsWithUri } from 'request';

export class ChatbotInsightiva implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Chatbot Insightiva',
    name: 'chatbotInsightiva',
    group: ['transform'],
    version: 1,
    description: 'Envia perguntas para o Chatbot da Insightiva via API interna',
    defaults: {
      name: 'Chatbot Insightiva',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Pergunta',
        name: 'pergunta',
        type: 'string',
        default: '',
        placeholder: 'Digite a pergunta aqui',
        description: 'Texto da pergunta para o Chatbot Insightiva',
      },
    ],
  };

  async execute(this: IExecuteFunctions) {
    const items = this.getInputData();
    const returnData = [];

    for (let i = 0; i < items.length; i++) {
      const pergunta = this.getNodeParameter('pergunta', i) as string;

      const options: OptionsWithUri = {
        method: 'POST',
        uri: 'http://api:8000/chat',
        body: {
          pergunta,
        },
        json: true,
      };

      try {
        const responseData = await this.helpers.request(options);
        returnData.push({ json: responseData });
      } catch (error) {
        throw new Error(`Erro ao chamar Chatbot Insightiva: ${error.message}`);
      }
    }

    return this.prepareOutputData(returnData);
  }
}
EOF

echo "🔗 Registrando node no index..."
echo "export * from './${NODE_DIR}/${NODE_DIR}.node';" >> packages/nodes-base/nodes/index.ts

echo "🔧 Instalando dependências com pnpm..."
pnpm install

echo "✅ Tudo pronto! Você pode agora rodar o n8n com:"
echo ""
echo "  cd $PROJETO"
echo "  pnpm run dev"
echo ""
