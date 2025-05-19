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
