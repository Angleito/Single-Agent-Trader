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

interface BraveSearchResponse {
	web: {
		results: Array<{
			title: string;
			url: string;
			description: string;
		}>;
	};
}

export class BraveSearchProvider implements SearchProvider {
	name = 'brave';
	description =
		'Privacy-focused search engine with good coverage of technical topics. Features native support for search operators (site:, filetype:, intitle:, inurl:, before:, after:, and exact phrases). Best for technical documentation, developer resources, and privacy-sensitive queries.';

	async search(params: BaseSearchParams): Promise<SearchResult[]> {
		const api_key = validate_api_key(
			config.search.brave.api_key,
			this.name,
		);

		// Parse search operators from the query
		const parsed_query = parse_search_operators(params.query);
		const search_params = apply_search_operators(parsed_query);

		const search_request = async () => {
			try {
				let query = sanitize_query(search_params.query);

				// Build operator filters
				const filters: string[] = [];

				// Handle domain filters
				const include_domains = [
					...(params.include_domains ?? []),
					...(search_params.include_domains ?? []),
				];
				if (include_domains.length) {
					const domain_filter = include_domains
						.map((domain) => `site:${domain}`)
						.join(' OR ');
					filters.push(`(${domain_filter})`);
				}

				const exclude_domains = [
					...(params.exclude_domains ?? []),
					...(search_params.exclude_domains ?? []),
				];
				if (exclude_domains.length) {
					filters.push(
						...exclude_domains.map((domain) => `-site:${domain}`),
					);
				}

				// Add file type filter
				if (search_params.file_type) {
					filters.push(`filetype:${search_params.file_type}`);
				}

				// Add title filter
				if (search_params.title_filter) {
					filters.push(`intitle:${search_params.title_filter}`);
				}

				// Add URL filter
				if (search_params.url_filter) {
					filters.push(`inurl:${search_params.url_filter}`);
				}

				// Add date filters
				if (search_params.date_before) {
					filters.push(`before:${search_params.date_before}`);
				}
				if (search_params.date_after) {
					filters.push(`after:${search_params.date_after}`);
				}

				// Add exact phrases
				if (search_params.exact_phrases?.length) {
					filters.push(
						...search_params.exact_phrases.map(
							(phrase) => `"${phrase}"`,
						),
					);
				}

				// Combine query with filters
				if (filters.length > 0) {
					query = `${query} ${filters.join(' ')}`;
				}

				const query_params = new URLSearchParams({
					q: query,
					count: (params.limit ?? 10).toString(),
				});

				const response = await fetch(
					`${config.search.brave.base_url}/web/search?${query_params}`,
					{
						method: 'GET',
						headers: {
							Accept: 'application/json',
							'X-Subscription-Token': api_key,
						},
						signal: AbortSignal.timeout(config.search.brave.timeout),
					},
				);

				let data: BraveSearchResponse & { message?: string };
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

				if (!response.ok || !data.web?.results) {
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
								'Brave Search API internal error',
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

				return data.web.results.map((result) => ({
					title: result.title,
					url: result.url,
					snippet: result.description,
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
