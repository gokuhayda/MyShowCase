import {
	ICredentialType,
	NodePropertyTypes,
} from 'n8n-workflow';

export class ChatbotInsightivaApi implements ICredentialType {
	name = 'chatbotInsightivaApi';
	displayName = 'Chatbot Insightiva API';
	properties = [
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string' as NodePropertyTypes,
			default: '',
		},
	];
}
