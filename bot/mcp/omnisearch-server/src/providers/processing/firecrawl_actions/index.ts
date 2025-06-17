import {
	ErrorType,
	ProcessingProvider,
	ProcessingResult,
	ProviderError,
} from '../../../common/types.js';
import {
	is_valid_url,
	retry_with_backoff,
	validate_api_key,
} from '../../../common/utils.js';
import { config } from '../../../config/env.js';

interface FirecrawlActionsResponse {
	success: boolean;
	data?: {
		markdown?: string;
		html?: string;
		rawHtml?: string;
		screenshot?: string;
		actions?: {
			screenshots?: string[];
		};
		metadata?: {
			title?: string;
			description?: string;
			language?: string;
			sourceURL?: string;
			statusCode?: number;
			error?: string;
			[key: string]: any;
		};
	};
	error?: string;
}

// Define the action types
type ActionType = 'click' | 'type' | 'scroll' | 'wait' | 'select';

interface Action {
	type: ActionType;
	selector?: string;
	text?: string;
	x?: number;
	y?: number;
	duration?: number;
	value?: string;
}

export class FirecrawlActionsProvider implements ProcessingProvider {
	name = 'firecrawl_actions';
	description =
		'Support for page interactions (clicking, scrolling, etc.) before extraction for dynamic content using Firecrawl. Enables extraction from JavaScript-heavy sites, single-page applications, and content behind user interactions. Best for accessing content that requires navigation, form filling, or other interactions.';

	async process_content(
		url: string | string[],
		extract_depth: 'basic' | 'advanced' = 'basic',
	): Promise<ProcessingResult> {
		// Actions works with a single URL
		const actions_url = Array.isArray(url) ? url[0] : url;

		// Validate URL
		if (!is_valid_url(actions_url)) {
			throw new ProviderError(
				ErrorType.INVALID_INPUT,
				`Invalid URL provided: ${actions_url}`,
				this.name,
			);
		}

		const actions_request = async () => {
			const api_key = validate_api_key(
				config.processing.firecrawl_actions.api_key,
				this.name,
			);

			try {
				// Define actions based on extract_depth
				// For basic, we'll just scroll down once to load more content
				// For advanced, we'll perform more complex interactions
				const actions: Action[] = extract_depth === 'advanced'
					? [
							{ type: 'wait', duration: 2000 }, // Wait for initial page load
							{ type: 'scroll', duration: 1000 }, // Scroll down
							{ type: 'wait', duration: 1000 }, // Wait for content to load
							{ type: 'scroll', duration: 1000 }, // Scroll down more
							{ type: 'wait', duration: 1000 }, // Wait for content to load
							// Click on "Read more" or "Show more" buttons if they exist
							{ type: 'click', selector: 'button:contains("Read more"), button:contains("Show more"), a:contains("Read more"), a:contains("Show more")' },
							{ type: 'wait', duration: 2000 }, // Wait for content to expand
					  ]
					: [
							{ type: 'wait', duration: 2000 }, // Wait for initial page load
							{ type: 'scroll', duration: 1000 }, // Scroll down once
							{ type: 'wait', duration: 1000 }, // Wait for content to load
					  ];

				// Start the actions
				const actions_response = await fetch(
					config.processing.firecrawl_actions.base_url,
					{
						method: 'POST',
						headers: {
							'Authorization': `Bearer ${api_key}`,
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							url: actions_url,
							formats: ['markdown', 'screenshot'], // Prefer markdown for LLM consumption and include screenshot
							actions: actions.map(action => {
								// Convert our action format to Firecrawl's action format
								switch (action.type) {
									case 'wait':
										return {
											type: 'wait',
											milliseconds: action.duration || 1000,
											selector: action.selector,
										};
									case 'scroll':
										return {
											type: 'scroll',
											// Firecrawl might use different parameters for scroll
											// Adjust as needed based on their documentation
										};
									case 'click':
										return {
											type: 'click',
											selector: action.selector,
											x: action.x,
											y: action.y,
										};
									case 'type':
										return {
											type: 'type',
											selector: action.selector,
											text: action.text || '',
										};
									case 'select':
										return {
											type: 'select',
											selector: action.selector,
											value: action.value || '',
										};
									default:
										return action;
								}
							}),
						}),
						signal: AbortSignal.timeout(
							config.processing.firecrawl_actions.timeout,
						),
					},
				);

				if (!actions_response.ok) {
					// Handle error responses based on status codes
					switch (actions_response.status) {
						case 400:
							throw new ProviderError(
								ErrorType.INVALID_INPUT,
								'Invalid request parameters',
								this.name,
							);
						case 401:
							throw new ProviderError(
								ErrorType.API_ERROR,
								'Invalid API key',
								this.name,
							);
						case 403:
							throw new ProviderError(
								ErrorType.API_ERROR,
								'API key does not have access to this endpoint',
								this.name,
							);
						case 429:
							throw new ProviderError(
								ErrorType.RATE_LIMIT,
								'Rate limit exceeded',
								this.name,
							);
						case 500:
							throw new ProviderError(
								ErrorType.PROVIDER_ERROR,
								'Firecrawl API internal error',
								this.name,
							);
						default:
							throw new ProviderError(
								ErrorType.API_ERROR,
								`Unexpected error: ${actions_response.statusText}`,
								this.name,
							);
					}
				}

				const actions_data = (await actions_response.json()) as FirecrawlActionsResponse;

				// Check if there was an error in the response
				if (!actions_data.success || actions_data.error) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						`Error performing actions: ${actions_data.error || 'Unknown error'}`,
						this.name,
					);
				}

				// Check if we have data
				if (!actions_data.data) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'No data returned from API',
						this.name,
					);
				}

				// Check if we have content
				if (!actions_data.data.markdown && !actions_data.data.html && !actions_data.data.rawHtml) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'No content extracted after performing actions',
						this.name,
					);
				}

				// Prefer markdown, fallback to HTML, then rawHtml
				const content = actions_data.data.markdown || actions_data.data.html || actions_data.data.rawHtml || '';

				// Add information about the actions performed
				const actions_description = `# Content from ${actions_url} after interactions\n\n` +
					`The following actions were performed before extraction:\n\n` +
					actions.map((action, index) => {
						switch (action.type) {
							case 'click':
								return `${index + 1}. Click on ${action.selector || `coordinates (${action.x}, ${action.y})`}`;
							case 'type':
								return `${index + 1}. Type "${action.text}" ${action.selector ? `into ${action.selector}` : ''}`;
							case 'scroll':
								return `${index + 1}. Scroll ${action.duration ? `for ${action.duration}ms` : ''}`;
							case 'wait':
								return `${index + 1}. Wait ${action.duration ? `for ${action.duration}ms` : ''}`;
							case 'select':
								return `${index + 1}. Select "${action.value}" from ${action.selector}`;
							default:
								return `${index + 1}. Perform ${action.type} action`;
						}
					}).join('\n') +
					'\n\n---\n\n' +
					content;

				// Create a single raw_content entry
				const raw_contents = [{
					url: actions_url,
					content: actions_description,
				}];

				// Calculate word count
				const word_count = actions_description
					.split(/\s+/)
					.filter(Boolean).length;

				return {
					content: actions_description,
					raw_contents,
					metadata: {
						title: `Content from ${actions_url} after interactions`,
						word_count,
						urls_processed: 1,
						successful_extractions: 1,
						extract_depth,
						screenshot: actions_data.data.screenshot,
					},
					source_provider: this.name,
				};
			} catch (error) {
				if (error instanceof ProviderError) {
					throw error;
				}
				throw new ProviderError(
					ErrorType.API_ERROR,
					`Failed to perform actions: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(actions_request);
	}
}
