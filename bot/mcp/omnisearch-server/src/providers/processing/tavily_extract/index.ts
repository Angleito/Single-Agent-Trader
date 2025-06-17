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

interface TavilyExtractResponse {
	results: {
		url: string;
		raw_content: string;
		images?: {
			url: string;
			alt_text?: string;
		}[];
	}[];
	failed_results: string[];
	response_time: number;
}

export class TavilyExtractProvider implements ProcessingProvider {
	name = 'tavily_extract';
	description =
		'Extract web page content from single or multiple URLs using Tavily Extract. Efficiently converts web content into clean, processable text with configurable extraction depth and optional image extraction. Returns both combined and individual URL content. Best for content analysis, data collection, and research.';

	async process_content(
		url: string | string[],
		extract_depth: 'basic' | 'advanced' = 'basic',
	): Promise<ProcessingResult> {
		const urls = Array.isArray(url) ? url : [url];

		// Validate all URLs
		for (const u of urls) {
			if (!is_valid_url(u)) {
				throw new ProviderError(
					ErrorType.INVALID_INPUT,
					`Invalid URL provided: ${u}`,
					this.name,
				);
			}
		}

		const extract_request = async () => {
			const api_key = validate_api_key(
				config.processing.tavily_extract.api_key,
				this.name,
			);

			try {
				const response = await fetch(
					`${config.processing.tavily_extract.base_url}/extract`,
					{
						method: 'POST',
						headers: {
							Authorization: `Bearer ${api_key}`,
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							urls: urls,
							include_images: false,
							extract_depth,
						}),
						signal: AbortSignal.timeout(
							config.processing.tavily_extract.timeout,
						),
					},
				);

				if (!response.ok) {
					// Handle error responses based on status codes
					switch (response.status) {
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
								'Tavily Extract API internal error',
								this.name,
							);
						default:
							throw new ProviderError(
								ErrorType.API_ERROR,
								`Unexpected error: ${response.statusText}`,
								this.name,
							);
					}
				}

				const data = (await response.json()) as TavilyExtractResponse;

				// Check if there are any results
				if (data.results.length === 0) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'No content extracted from URL',
						this.name,
					);
				}

				// Map results to raw_contents array
				const raw_contents = data.results.map((result) => ({
					url: result.url,
					content: result.raw_content,
				}));

				// Combine all results into a single content string
				const combined_content = raw_contents
					.map((result) => result.content)
					.join('\n\n');

				// Calculate total word count
				const word_count = combined_content
					.split(/\s+/)
					.filter(Boolean).length;

				// Include any failed URLs in metadata
				const failed_urls =
					data.failed_results.length > 0
						? data.failed_results
						: undefined;

				return {
					content: combined_content,
					raw_contents,
					metadata: {
						word_count,
						failed_urls,
						urls_processed: urls.length,
						successful_extractions: data.results.length,
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
					`Failed to extract content: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(extract_request);
	}
}
