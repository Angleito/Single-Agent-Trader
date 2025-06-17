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

interface FirecrawlScrapeResponse {
	success: boolean;
	data?: {
		markdown?: string;
		html?: string;
		rawHtml?: string;
		screenshot?: string;
		links?: string[];
		metadata?: {
			title?: string;
			description?: string;
			language?: string;
			sourceURL?: string;
			statusCode?: number;
			error?: string;
			[key: string]: any;
		};
		llm_extraction?: any;
		warning?: string;
	};
	error?: string;
}

export class FirecrawlScrapeProvider implements ProcessingProvider {
	name = 'firecrawl_scrape';
	description =
		'Extract clean, LLM-ready data from single URLs with enhanced formatting options using Firecrawl. Efficiently converts web content into markdown, plain text, or structured data with configurable extraction options. Best for content analysis, data collection, and AI training data preparation.';

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

		const scrape_request = async () => {
			const api_key = validate_api_key(
				config.processing.firecrawl_scrape.api_key,
				this.name,
			);

			try {
				// Process each URL and collect results
				const results = await Promise.all(
					urls.map(async (single_url) => {
						try {
							const response = await fetch(
								config.processing.firecrawl_scrape.base_url,
								{
									method: 'POST',
									headers: {
										Authorization: `Bearer ${api_key}`,
										'Content-Type': 'application/json',
									},
									body: JSON.stringify({
										url: single_url,
										formats: ['markdown'], // Prefer markdown for LLM consumption
										onlyMainContent: true,
										// Use advanced options if extract_depth is advanced
										waitFor:
											extract_depth === 'advanced' ? 5000 : 2000,
									}),
									signal: AbortSignal.timeout(
										config.processing.firecrawl_scrape.timeout,
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
											'Firecrawl API internal error',
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

							const data =
								(await response.json()) as FirecrawlScrapeResponse;

							// Check if there was an error in the response
							if (!data.success || data.error) {
								throw new ProviderError(
									ErrorType.PROVIDER_ERROR,
									`Error scraping URL: ${
										data.error || 'Unknown error'
									}`,
									this.name,
								);
							}

							// Check if we have data
							if (!data.data) {
								throw new ProviderError(
									ErrorType.PROVIDER_ERROR,
									'No data returned from API',
									this.name,
								);
							}

							// Check if content was successfully extracted
							if (!data.data.markdown && !data.data.html && !data.data.rawHtml) {
								throw new ProviderError(
									ErrorType.PROVIDER_ERROR,
									'No content extracted from URL',
									this.name,
								);
							}

							// Prefer markdown, fallback to HTML, then rawHtml
							const content =
								data.data.markdown || data.data.html || data.data.rawHtml || '';

							return {
								url: single_url,
								content,
								metadata: data.data.metadata,
								success: true,
							};
						} catch (error) {
							// Log the error but continue processing other URLs
							console.error(`Error processing ${single_url}:`, error);
							return {
								url: single_url,
								content: '',
								success: false,
								error:
									error instanceof Error
										? error.message
										: 'Unknown error',
							};
						}
					}),
				);

				// Filter successful and failed results
				const successful_results = results.filter((r) => r.success);
				const failed_urls = results
					.filter((r) => !r.success)
					.map((r) => r.url);

				// If all URLs failed, throw an error
				if (successful_results.length === 0) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'Failed to extract content from all URLs',
						this.name,
					);
				}

				// Map results to raw_contents array
				const raw_contents = successful_results.map((result) => ({
					url: result.url,
					content: result.content,
				}));

				// Combine all results into a single content string
				const combined_content = raw_contents
					.map((result) => result.content)
					.join('\n\n');

				// Calculate total word count
				const word_count = combined_content
					.split(/\s+/)
					.filter(Boolean).length;

				// Get title from first successful result if available
				const title = successful_results[0]?.metadata?.title;

				return {
					content: combined_content,
					raw_contents,
					metadata: {
						title,
						word_count,
						failed_urls:
							failed_urls.length > 0 ? failed_urls : undefined,
						urls_processed: urls.length,
						successful_extractions: successful_results.length,
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

		return retry_with_backoff(scrape_request);
	}
}
