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

interface FirecrawlMapResponse {
	success: boolean;
	links?: string[];
	error?: string;
}

export class FirecrawlMapProvider implements ProcessingProvider {
	name = 'firecrawl_map';
	description =
		'Fast URL collection from websites for comprehensive site mapping using Firecrawl. Efficiently discovers all accessible URLs within a domain without extracting content. Best for site auditing, URL discovery, and preparing for targeted content extraction.';

	async process_content(
		url: string | string[],
		extract_depth: 'basic' | 'advanced' = 'basic',
	): Promise<ProcessingResult> {
		// Map only works with a single URL (the starting point)
		const map_url = Array.isArray(url) ? url[0] : url;

		// Validate URL
		if (!is_valid_url(map_url)) {
			throw new ProviderError(
				ErrorType.INVALID_INPUT,
				`Invalid URL provided: ${map_url}`,
				this.name,
			);
		}

		const map_request = async () => {
			const api_key = validate_api_key(
				config.processing.firecrawl_map.api_key,
				this.name,
			);

			try {
				// Start the map operation
				const map_response = await fetch(
					config.processing.firecrawl_map.base_url,
					{
						method: 'POST',
						headers: {
							'Authorization': `Bearer ${api_key}`,
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							url: map_url,
							// Use advanced options if extract_depth is advanced
							limit: extract_depth === 'advanced' ? 200 : 50,
							ignoreSitemap: false, // Use sitemap for better coverage
							includeSubdomains: false, // Only include URLs from the same domain
						}),
						signal: AbortSignal.timeout(
							config.processing.firecrawl_map.timeout,
						),
					},
				);

				if (!map_response.ok) {
					// Handle error responses based on status codes
					switch (map_response.status) {
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
								`Unexpected error: ${map_response.statusText}`,
								this.name,
							);
					}
				}

				const map_data = (await map_response.json()) as FirecrawlMapResponse;

				// Check if there was an error in the response
				if (!map_data.success || map_data.error) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						`Error mapping website: ${map_data.error || 'Unknown error'}`,
						this.name,
					);
				}

				// Check if we have links
				if (!map_data.links || map_data.links.length === 0) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'No URLs discovered during mapping',
						this.name,
					);
				}

				// Format the links as a list with descriptions
				const formatted_content = `# Site Map for ${map_url}\n\n` +
					`Found ${map_data.links.length} URLs:\n\n` +
					map_data.links.map((url) => `- ${url}`).join('\n');

				// Create a single raw_content entry with all URLs
				const raw_contents = [{
					url: map_url,
					content: formatted_content,
				}];

				return {
					content: formatted_content,
					raw_contents,
					metadata: {
						title: `Site Map for ${map_url}`,
						word_count: map_data.links.length, // Count URLs as "words"
						urls_processed: 1, // We only processed the starting URL
						successful_extractions: 1,
						extract_depth,
					},
					source_provider: this.name,
				};
			} catch (error) {
				if (error instanceof ProviderError) {
					throw error;
				}
				throw new ProviderError(
					ErrorType.API_ERROR,
					`Failed to map website: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(map_request);
	}
}
