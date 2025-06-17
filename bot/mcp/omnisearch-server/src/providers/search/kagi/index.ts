import {
	BaseSearchParams,
	ErrorType,
	ProviderError,
	SearchProvider,
	SearchResult,
} from '../../../common/types.js';
import {
	apply_search_operators,
	handle_rate_limit,
	parse_search_operators,
	retry_with_backoff,
	sanitize_query,
	validate_api_key,
} from '../../../common/utils.js';
import { config } from '../../../config/env.js';

interface KagiSearchResponse {
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

export class KagiSearchProvider implements SearchProvider {
	name = 'kagi';
	description =
		'High-quality search results with minimal advertising influence, focused on authoritative sources. Supports search operators in query string (site:, -site:, filetype:, intitle:, inurl:, before:, after:, and exact phrases). Features strong privacy protection and access to specialized knowledge indexes. Best for research, technical documentation, and finding high-quality content without SEO manipulation.';

	async search(params: BaseSearchParams): Promise<SearchResult[]> {
		const api_key = validate_api_key(
			config.search.kagi.api_key,
			this.name,
		);

		// Parse search operators from the query
		const parsed_query = parse_search_operators(params.query);
		const search_params = apply_search_operators(parsed_query);

		const search_request = async () => {
			try {
				let query = sanitize_query(search_params.query);
				const query_params = new URLSearchParams({
					q: query,
					limit: (params.limit ?? 10).toString(),
				});

				// Handle domain filters using query string operators
				const include_domains = [
					...(params.include_domains ?? []),
					...(search_params.include_domains ?? []),
				];
				if (include_domains.length) {
					const domain_filter = include_domains
						.map((domain) => `site:${domain}`)
						.join(' OR ');
					query = `${query} (${domain_filter})`;
				}

				const exclude_domains = [
					...(params.exclude_domains ?? []),
					...(search_params.exclude_domains ?? []),
				];
				if (exclude_domains.length) {
					query = `${query} ${exclude_domains
						.map((domain) => `-site:${domain}`)
						.join(' ')}`;
				}

				// Update query parameter with domain filters
				query_params.set('q', query);

				// Add file type filter
				if (search_params.file_type) {
					query_params.append('file_type', search_params.file_type);
				}

				// Add time range filters
				if (search_params.date_before || search_params.date_after) {
					const time_range: string[] = [];
					if (search_params.date_after) {
						time_range.push(`after:${search_params.date_after}`);
					}
					if (search_params.date_before) {
						time_range.push(`before:${search_params.date_before}`);
					}
					query_params.append('time_range', time_range.join(','));
				}

				// Add title and URL filters to the query
				if (search_params.title_filter) {
					query += ` intitle:${search_params.title_filter}`;
					query_params.set('q', query);
				}
				if (search_params.url_filter) {
					query += ` inurl:${search_params.url_filter}`;
					query_params.set('q', query);
				}

				// Add exact phrases
				if (search_params.exact_phrases?.length) {
					query += ` ${search_params.exact_phrases
						.map((phrase) => `"${phrase}"`)
						.join(' ')}`;
					query_params.set('q', query);
				}

				const response = await fetch(
					`${config.search.kagi.base_url}/search?${query_params}`,
					{
						method: 'GET',
						headers: {
							Authorization: `Bot ${api_key}`,
							Accept: 'application/json',
						},
						signal: AbortSignal.timeout(config.search.kagi.timeout),
					},
				);

				let data: KagiSearchResponse & { message?: string };
				try {
					const text = await response.text();
					data = JSON.parse(text);
				} catch (error) {
					throw new ProviderError(
						ErrorType.API_ERROR,
						`Invalid JSON response: ${
							error instanceof Error ? error.message : 'Unknown error'
						}`,
						this.name,
					);
				}

				if (!response.ok || !data.data) {
					const error_message = data.message || response.statusText;
					switch (response.status) {
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
								'Kagi Search API internal error',
								this.name,
							);
						default:
							throw new ProviderError(
								ErrorType.API_ERROR,
								`Unexpected error: ${error_message}`,
								this.name,
							);
					}
				}

				return data.data.map((result) => ({
					title: result.title,
					url: result.url,
					snippet: result.snippet,
					score: result.rank,
					source_provider: this.name,
				}));
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

		return retry_with_backoff(search_request);
	}
}
