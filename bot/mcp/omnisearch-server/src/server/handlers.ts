import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
	ListResourcesRequestSchema,
	ListResourceTemplatesRequestSchema,
	ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { available_providers } from './tools.js';

export const setup_handlers = (server: Server) => {
	// List available resources
	server.setRequestHandler(ListResourcesRequestSchema, async () => ({
		resources: [
			{
				uri: 'omnisearch://providers/status',
				name: 'Provider Status',
				mimeType: 'application/json',
				description: 'Current status of all search providers',
			},
		],
	}));

	// List resource templates
	server.setRequestHandler(
		ListResourceTemplatesRequestSchema,
		async () => ({
			resourceTemplates: [
				{
					uriTemplate: 'omnisearch://search/{provider}/info',
					name: 'Search Provider Info',
					mimeType: 'application/json',
					description: 'Information about a specific search provider',
				},
			],
		}),
	);

	// Handle resource reads
	server.setRequestHandler(
		ReadResourceRequestSchema,
		async (request) => {
			const { uri } = request.params;

			// Handle provider status resource
			if (uri === 'omnisearch://providers/status') {
				return {
					contents: [
						{
							uri,
							mimeType: 'application/json',
							text: JSON.stringify(
								{
									status: 'operational',
									providers: {
										search: Array.from(available_providers.search),
										ai_response: Array.from(
											available_providers.ai_response,
										),
										processing: Array.from(
											available_providers.processing,
										),
										enhancement: Array.from(
											available_providers.enhancement,
										),
									},
									available_count: {
										search: available_providers.search.size,
										ai_response: available_providers.ai_response.size,
										processing: available_providers.processing.size,
										enhancement: available_providers.enhancement.size,
										total:
											available_providers.search.size +
											available_providers.ai_response.size +
											available_providers.processing.size +
											available_providers.enhancement.size,
									},
								},
								null,
								2,
							),
						},
					],
				};
			}

			// Handle provider info template
			const providerMatch = uri.match(
				/^omnisearch:\/\/search\/([^/]+)\/info$/,
			);
			if (providerMatch) {
				const providerName = providerMatch[1];

				// Check if provider is available
				const isAvailable =
					available_providers.search.has(providerName) ||
					available_providers.ai_response.has(providerName);

				if (!isAvailable) {
					throw new Error(
						`Provider not available: ${providerName} (missing API key)`,
					);
				}

				return {
					contents: [
						{
							uri,
							mimeType: 'application/json',
							text: JSON.stringify(
								{
									name: providerName,
									status: 'active',
									capabilities: ['web_search', 'news_search'],
									rate_limits: {
										requests_per_minute: 60,
										requests_per_day: 1000,
									},
								},
								null,
								2,
							),
						},
					],
				};
			}

			throw new Error(`Unknown resource URI: ${uri}`);
		},
	);
};
