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

interface FirecrawlExtractResponse {
	success: boolean;
	id: string;
	error?: string;
}

interface FirecrawlExtractStatusResponse {
	success: boolean;
	id: string;
	status: string;
	data?: any;
	error?: string;
}

export class FirecrawlExtractProvider implements ProcessingProvider {
	name = 'firecrawl_extract';
	description =
		'Structured data extraction with AI using natural language prompts via Firecrawl. Extracts specific information from web pages based on custom extraction instructions. Best for targeted data collection, information extraction, and converting unstructured web content into structured data.';

	async process_content(
		url: string | string[],
		extract_depth: 'basic' | 'advanced' = 'basic',
	): Promise<ProcessingResult> {
		// Extract works with a single URL at a time
		const extract_url = Array.isArray(url) ? url[0] : url;

		// Validate URL
		if (!is_valid_url(extract_url)) {
			throw new ProviderError(
				ErrorType.INVALID_INPUT,
				`Invalid URL provided: ${extract_url}`,
				this.name,
			);
		}

		const extract_request = async () => {
			const api_key = validate_api_key(
				config.processing.firecrawl_extract.api_key,
				this.name,
			);

			try {
				// Define extraction instructions based on extract_depth
				const extraction_prompt = extract_depth === 'advanced'
					? 'Extract all relevant information from this page including: title, author, date published, main content, categories or tags, related links, and any structured data like product information, pricing, or specifications. Format the data in a well-structured way.'
					: 'Extract the main content, title, and author from this page. Summarize the key information.';

				// Start the extraction
				const extract_response = await fetch(
					config.processing.firecrawl_extract.base_url,
					{
						method: 'POST',
						headers: {
							'Authorization': `Bearer ${api_key}`,
							'Content-Type': 'application/json',
						},
						body: JSON.stringify({
							urls: [extract_url],
							prompt: extraction_prompt,
							showSources: true,
							scrapeOptions: {
								formats: ['markdown'],
								onlyMainContent: true,
								waitFor: extract_depth === 'advanced' ? 5000 : 2000,
							},
						}),
						signal: AbortSignal.timeout(
							config.processing.firecrawl_extract.timeout,
						),
					},
				);

				if (!extract_response.ok) {
					// Handle error responses based on status codes
					switch (extract_response.status) {
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
								`Unexpected error: ${extract_response.statusText}`,
								this.name,
							);
					}
				}

				const extract_data = (await extract_response.json()) as FirecrawlExtractResponse;

				// Check if there was an error in the response
				if (!extract_data.success || extract_data.error) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						`Error starting extraction: ${extract_data.error || 'Unknown error'}`,
						this.name,
					);
				}

				// For extractions, we always need to poll for results
				const extract_id = extract_data.id;
				let status_data: FirecrawlExtractStatusResponse | null = null;
				let attempts = 0;
				const max_attempts = 15; // More attempts for extraction as it can take longer

				// Poll for results
				while (attempts < max_attempts) {
					attempts++;
					await new Promise((resolve) => setTimeout(resolve, 3000)); // Wait 3 seconds between polls

					const status_response = await fetch(
						`${config.processing.firecrawl_extract.base_url}/${extract_id}`,
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

					const status_result = (await status_response.json()) as FirecrawlExtractStatusResponse;

					if (!status_result.success) {
						throw new ProviderError(
							ErrorType.PROVIDER_ERROR,
							`Error checking extraction status: ${status_result.error || 'Unknown error'}`,
							this.name,
						);
					}

					if (status_result.status === 'completed' && status_result.data) {
						status_data = status_result;
						break;
					} else if (status_result.status === 'error') {
						throw new ProviderError(
							ErrorType.PROVIDER_ERROR,
							`Error extracting data: ${status_result.error || 'Unknown error'}`,
							this.name,
						);
					}

					// If still processing, continue polling
				}

				// If we've reached max attempts without completion
				if (!status_data || !status_data.data) {
					throw new ProviderError(
						ErrorType.PROVIDER_ERROR,
						'No data extracted from URL',
						this.name,
					);
				}

				// Format the extracted data as markdown
				let formatted_content = `# Extracted Data from ${extract_url}\n\n`;

				// Add each extracted field
				for (const [key, value] of Object.entries(status_data.data)) {
					if (typeof value === 'string') {
						formatted_content += `## ${key.charAt(0).toUpperCase() + key.slice(1)}\n\n${value}\n\n`;
					} else if (Array.isArray(value)) {
						formatted_content += `## ${key.charAt(0).toUpperCase() + key.slice(1)}\n\n`;
						value.forEach((item, index) => {
							if (typeof item === 'object') {
								formatted_content += `### Item ${index + 1}\n\n`;
								for (const [itemKey, itemValue] of Object.entries(item)) {
									formatted_content += `- **${itemKey}**: ${itemValue}\n`;
								}
								formatted_content += '\n';
							} else {
								formatted_content += `- ${item}\n`;
							}
						});
						formatted_content += '\n';
					} else if (typeof value === 'object' && value !== null) {
						formatted_content += `## ${key.charAt(0).toUpperCase() + key.slice(1)}\n\n`;
						for (const [subKey, subValue] of Object.entries(value)) {
							formatted_content += `- **${subKey}**: ${subValue}\n`;
						}
						formatted_content += '\n';
					}
				}

				// Create a single raw_content entry
				const raw_contents = [{
					url: extract_url,
					content: formatted_content,
				}];

				// Get title if available
				const title = status_data.data.title || `Extracted Data from ${extract_url}`;

				// Count words in the formatted content
				const word_count = formatted_content
					.split(/\s+/)
					.filter(Boolean).length;

				return {
					content: formatted_content,
					raw_contents,
					metadata: {
						title,
						word_count,
						urls_processed: 1,
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
					`Failed to extract data: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(extract_request);
	}
}
