import {
	EnhancementProvider,
	EnhancementResult,
	ErrorType,
	ProviderError,
} from '../../../common/types.js';
import {
	handle_rate_limit,
	retry_with_backoff,
	sanitize_query,
	validate_api_key,
} from '../../../common/utils.js';
import { config } from '../../../config/env.js';

export interface EnrichmentResponse {
	data: Array<{
		title: string;
		url: string;
		snippet: string;
		rank?: number;
	}>;
	meta?: {
		total_hits: number;
		api_balance?: number;
	};
}

export class KagiEnrichmentProvider implements EnhancementProvider {
	name = 'kagi_enrichment';
	description =
		'Provides supplementary content from specialized indexes (Teclis for web, TinyGem for news). Ideal for discovering non-mainstream results and enriching content with specialized knowledge.';

	async enhance_content(content: string): Promise<EnhancementResult> {
		const api_key = validate_api_key(
			config.enhancement.kagi_enrichment.api_key,
			this.name,
		);

		const enrich_request = async () => {
			try {
				// Try both web and news endpoints
				const results = await Promise.all([
					fetch(
						`https://kagi.com/api/v0/enrich/web?${new URLSearchParams({
							q: sanitize_query('artificial intelligence software development'),
							limit: '5',
						})}`,
						{
							method: 'GET',
							headers: {
								Authorization: `Bot ${api_key}`,
								Accept: 'application/json',
							},
							signal: AbortSignal.timeout(
								config.enhancement.kagi_enrichment.timeout,
							),
						},
					),
					fetch(
						`https://kagi.com/api/v0/enrich/news?${new URLSearchParams({
							q: sanitize_query('artificial intelligence code generation testing'),
							limit: '5',
						})}`,
						{
							method: 'GET',
							headers: {
								Authorization: `Bot ${api_key}`,
								Accept: 'application/json',
							},
							signal: AbortSignal.timeout(
								config.enhancement.kagi_enrichment.timeout,
							),
						},
					),
				]);

				const [webResponse, newsResponse] = results;

				// Parse and validate responses
				let webData: EnrichmentResponse & { message?: string };
				let newsData: EnrichmentResponse & { message?: string };

				try {
					const webText = await webResponse.text();
					webData = JSON.parse(webText);
				} catch (error) {
					throw new ProviderError(
						ErrorType.API_ERROR,
						`Invalid JSON response from web endpoint: ${
							error instanceof Error ? error.message : 'Unknown error'
						}`,
						this.name,
					);
				}

				try {
					const newsText = await newsResponse.text();
					newsData = JSON.parse(newsText);
				} catch (error) {
					throw new ProviderError(
						ErrorType.API_ERROR,
						`Invalid JSON response from news endpoint: ${
							error instanceof Error ? error.message : 'Unknown error'
						}`,
						this.name,
					);
				}

				if (!webResponse.ok || !webData.data) {
					const error_message = webData.message || webResponse.statusText;
					switch (webResponse.status) {
						case 401:
							throw new ProviderError(
								ErrorType.API_ERROR,
								'Invalid API key',
								this.name,
							);
						case 429:
							handle_rate_limit(this.name);
						case 500:
							throw new ProviderError(
								ErrorType.PROVIDER_ERROR,
								'Kagi Enrichment API internal error',
								this.name,
							);
						default:
							throw new ProviderError(
								ErrorType.API_ERROR,
								`Unexpected error from web endpoint: ${error_message}`,
								this.name,
							);
					}
				}

				if (!newsResponse.ok || !newsData.data) {
					const error_message = newsData.message || newsResponse.statusText;
					switch (newsResponse.status) {
						case 401:
							throw new ProviderError(
								ErrorType.API_ERROR,
								'Invalid API key',
								this.name,
							);
						case 429:
							handle_rate_limit(this.name);
						case 500:
							throw new ProviderError(
								ErrorType.PROVIDER_ERROR,
								'Kagi Enrichment API internal error',
								this.name,
							);
						default:
							throw new ProviderError(
								ErrorType.API_ERROR,
								`Unexpected error from news endpoint: ${error_message}`,
								this.name,
							);
					}
				}

				// Combine and filter results
				const allData = [...webData.data, ...newsData.data]
					.filter(result => 
						// Filter for results about software/development/AI
						result.snippet?.toLowerCase().includes('software') ||
						result.snippet?.toLowerCase().includes('develop') ||
						result.snippet?.toLowerCase().includes('programming') ||
						result.snippet?.toLowerCase().includes('code') ||
						result.snippet?.toLowerCase().includes('artificial intelligence') ||
						result.snippet?.toLowerCase().includes('ai')
					);

				// Clean and combine snippets
				const enhanced_content = allData
					.map((result) => result.snippet)
					.filter(Boolean)
					.map(snippet => 
						// Fix HTML entities
						snippet
							.replace(/&#39;/g, "'")
							.replace(/&quot;/g, '"')
							.replace(/&amp;/g, '&')
							.replace(/&lt;/g, '<')
							.replace(/&gt;/g, '>')
					)
					.join('\n\n');

				return {
					original_content: content,
					enhanced_content,
					enhancements: [
						{
							type: 'content_enrichment',
							description:
								'Added supplementary information from Teclis (web) and TinyGem (news) specialized indexes',
						},
					],
					sources: allData.map((result) => ({
						title: result.title,
						url: result.url,
					})),
					source_provider: this.name,
				};
			} catch (error) {
				if (error instanceof ProviderError) {
					throw error;
				}
				throw new ProviderError(
					ErrorType.API_ERROR,
					`Failed to fetch: ${
						error instanceof Error ? error.message : 'Unknown error'
					}`,
					this.name,
				);
			}
		};

		return retry_with_backoff(enrich_request);
	}
}
