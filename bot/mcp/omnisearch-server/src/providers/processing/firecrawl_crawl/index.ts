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

interface FirecrawlCrawlResponse {
	success: boolean;
	id: string;
	url: string;
	error?: string;
}

interface FirecrawlCrawlStatusResponse {
	success: boolean;
	id: string;
	status: string;
	total?: number;
	data?: Array<{
		url: string;
		markdown?: string;
		html?: string;
		rawHtml?: string;
		metadata?: {
			title?: string;
			description?: string;
			language?: string;
			sourceURL?: string;
			statusCode?: number;
			error?: string;
			[key: string]: any;
		};
		error?: string;
	}>;
	error?: string;
}

export class FirecrawlCrawlProvider implements ProcessingProvider {
	name = 'firecrawl_crawl';
	description =
		'Deep crawling of all accessible subpages on a website with configurable depth limits using Firecrawl. Efficiently discovers and extracts content from multiple pages within a domain. Best for comprehensive site analysis, content indexing, and data collection from entire websites.';

	async process_content(
		url: string | string[],
		extract_depth: 'basic' | 'advanced' = 'basic',
	): Promise<ProcessingResult> {
		// Crawl only works with a single URL (the starting point)
		const crawl_url = Array.isArray(url) ? url[0] : url;

		// Validate URL
		if (!is_valid_url(crawl_url)) {
			throw new ProviderError(
				ErrorType.INVALID_INPUT,
				`Invalid URL provided: ${crawl_url}`,
				this.name,
			);
		}

		const crawl_request = async () => {
			const api_key = validate_api_key(
				config.processing.firecrawl_crawl.api_key,
				this.name,
			);

			try {
				// Start the crawl
				const crawl_response = await fetch(
					config.processing.firecrawl_crawl.base_url,
					{
						method: 'POST',
						headers: {
							'Authorization': `Bearer ${api_key}`,
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							url: crawl_url,
							scrapeOptions: {
								formats: ['markdown'], // Prefer markdown for LLM consumption
								onlyMainContent: true,
							},
							// Use advanced options if extract_depth is advanced
							maxDepth: extract_depth === 'advanced' ? 3 : 1,
							limit: extract_depth === 'advanced' ? 50 : 20,
						}),
						signal: AbortSignal.timeout(
							config.processing.firecrawl_crawl.timeout,
						),
					},
				);

				if (!crawl_response.ok) {
					// Handle error responses based on status codes
					switch (crawl_response.status) {
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
								`Unexpected error: ${crawl_response.statusText}`,
								this.name,
							);
					}
				}

				const crawl_data = (await crawl_response.json()) as FirecrawlCrawlResponse;

				// Check if there was an error in the response
				if (!crawl_data.success || crawl_data.error) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						`Error starting crawl: ${crawl_data.error || 'Unknown error'}`,
						this.name,
					);
				}

				// For crawls, we always need to poll for results
				const crawl_id = crawl_data.id;
				let status_data: FirecrawlCrawlStatusResponse | null = null;
				let attempts = 0;
				const max_attempts = 20; // More attempts for crawling

				// Poll for results
				while (attempts < max_attempts) {
					attempts++;
					await new Promise((resolve) => setTimeout(resolve, 5000)); // Wait 5 seconds between polls

					const status_response = await fetch(
						`${config.processing.firecrawl_crawl.base_url}/${crawl_id}`,
						{
							method: 'GET',
							headers: {
								'Authorization': `Bearer ${api_key}`,
							},
							signal: AbortSignal.timeout(30000), // 30 second timeout for status checks
						},
					);

					if (!status_response.ok) {
						continue; // Skip this attempt if there's an error
					}

					const status_result = (await status_response.json()) as FirecrawlCrawlStatusResponse;

					if (!status_result.success) {
						throw new ProviderError(
							ErrorType.PROVIDER_ERROR,
							`Error checking crawl status: ${status_result.error || 'Unknown error'}`,
							this.name,
						);
					}

					if (status_result.status === 'completed' && status_result.data && status_result.data.length > 0) {
						status_data = status_result;
						break;
					} else if (status_result.status === 'error') {
						throw new ProviderError(
							ErrorType.PROVIDER_ERROR,
							`Error crawling website: ${status_result.error || 'Unknown error'}`,
							this.name,
						);
					}

					// If still processing, continue polling
				}

				// If we've reached max attempts without completion
				if (!status_data || !status_data.data || status_data.data.length === 0) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'Crawl timed out or returned no data - try again later or with a smaller scope',
						this.name,
					);
				}

				// Filter out failed pages
				const successful_pages = status_data.data.filter(
					(page) => !page.error && (page.markdown || page.html || page.rawHtml),
				);

				if (successful_pages.length === 0) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'All crawled pages failed to extract content',
						this.name,
					);
				}

				// Map results to raw_contents array
				const raw_contents = successful_pages.map((page) => ({
					url: page.url,
					content: page.markdown || page.html || page.rawHtml || '',
				}));

				// Combine all results into a single content string
				const combined_content = raw_contents
					.map(
						(result) =>
							`# ${result.url}\n\n${result.content}\n\n---\n\n`,
					)
					.join('\n\n');

				// Calculate total word count
				const word_count = combined_content
					.split(/\s+/)
					.filter(Boolean).length;

				// Get title from first successful result if available
				const title = successful_pages[0]?.metadata?.title;

				// Track failed URLs
				const failed_urls = status_data.data
					.filter((page) => page.error)
					.map((page) => page.url);

				return {
					content: combined_content,
					raw_contents,
					metadata: {
						title,
						word_count,
						failed_urls: failed_urls.length > 0 ? failed_urls : undefined,
						urls_processed: status_data.data.length,
						successful_extractions: successful_pages.length,
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
					`Failed to crawl website: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(crawl_request);
	}
}
