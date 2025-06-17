// Common utility functions for the MCP Omnisearch server

import { ErrorType, ProviderError } from './types.js';

export const validate_api_key = (
	key: string | undefined,
	provider: string,
): string => {
	if (!key) {
		throw new ProviderError(
			ErrorType.INVALID_INPUT,
			`API key not found for ${provider}`,
			provider,
		);
	}
	return key;
};

/**
 * Checks if an API key exists and is potentially valid without throwing an error
 * @param key The API key to check
 * @param provider The name of the provider (for logging purposes)
 * @returns boolean indicating if the key exists and is not empty
 */
export const is_api_key_valid = (
	key: string | undefined,
	provider: string,
): boolean => {
	if (!key || key.trim() === '') {
		console.warn(`API key not found or empty for ${provider}`);
		return false;
	}
	return true;
};

export const handle_rate_limit = (
	provider: string,
	reset_time?: Date,
): never => {
	throw new ProviderError(
		ErrorType.RATE_LIMIT,
		`Rate limit exceeded for ${provider}${
			reset_time ? `. Reset at ${reset_time.toISOString()}` : ''
		}`,
		provider,
		{ reset_time },
	);
};

export const sanitize_query = (query: string): string => {
	return query.trim().replace(/[\n\r]+/g, ' ');
};

export const create_error_response = (
	error: Error,
): { error: string } => {
	if (error instanceof ProviderError) {
		return {
			error: `${error.provider} error: ${error.message}`,
		};
	}
	return {
		error: `Unexpected error: ${error.message}`,
	};
};

export const merge_search_results = <T extends { score?: number }>(
	results: T[],
	limit?: number,
): T[] => {
	const sorted = [...results].sort((a, b) => {
		const score_a = a.score ?? 0;
		const score_b = b.score ?? 0;
		return score_b - score_a;
	});

	return limit ? sorted.slice(0, limit) : sorted;
};

export const extract_domain = (url: string): string => {
	try {
		const domain = new URL(url).hostname;
		return domain.startsWith('www.') ? domain.slice(4) : domain;
	} catch {
		return '';
	}
};

export const is_valid_url = (url: string): boolean => {
	try {
		new URL(url);
		return true;
	} catch {
		return false;
	}
};

export const delay = (ms: number): Promise<void> => {
	return new Promise((resolve) => setTimeout(resolve, ms));
};

export const retry_with_backoff = async <T>(
	fn: () => Promise<T>,
	max_retries: number = 3,
	initial_delay: number = 1000,
): Promise<T> => {
	let retries = 0;
	while (true) {
		try {
			return await fn();
		} catch (error) {
			if (retries >= max_retries) {
				throw error;
			}
			const delay_time = initial_delay * Math.pow(2, retries);
			await delay(delay_time);
			retries++;
		}
	}
};

export interface SearchOperator {
	type:
		| 'site'
		| 'exclude_site'
		| 'filetype'
		| 'intitle'
		| 'inurl'
		| 'before'
		| 'after'
		| 'exact'
		| 'boolean';
	value: string;
	original_text: string;
}

export interface ParsedQuery {
	base_query: string;
	operators: SearchOperator[];
}

const operator_patterns = {
	site: /site:([^\s]+)/g,
	exclude_site: /-site:([^\s]+)/g,
	filetype: /filetype:([^\s]+)/g,
	intitle: /intitle:([^\s]+)/g,
	inurl: /inurl:([^\s]+)/g,
	before: /before:(\d{4}(?:-\d{2}(?:-\d{2})?)?)/g,
	after: /after:(\d{4}(?:-\d{2}(?:-\d{2})?)?)/g,
	exact: /"([^"]+)"/g,
	boolean: /\b(AND|OR|NOT)\b/g,
};

export const parse_search_operators = (
	query: string,
): ParsedQuery => {
	const operators: SearchOperator[] = [];
	let modified_query = query;

	// Extract operators
	Object.entries(operator_patterns).forEach(([type, pattern]) => {
		modified_query = modified_query.replace(
			pattern,
			(match, value) => {
				operators.push({
					type: type as SearchOperator['type'],
					value: value,
					original_text: match,
				});
				return '';
			},
		);
	});

	// Clean up the base query
	const base_query = modified_query.replace(/\s+/g, ' ').trim();

	return {
		base_query,
		operators,
	};
};

export interface SearchParams {
	query: string;
	include_domains?: string[];
	exclude_domains?: string[];
	file_type?: string;
	title_filter?: string;
	url_filter?: string;
	date_before?: string;
	date_after?: string;
	exact_phrases?: string[];
	boolean_operators?: {
		type: 'AND' | 'OR' | 'NOT';
		terms: string[];
	}[];
}

export const apply_search_operators = (
	parsed_query: ParsedQuery,
): SearchParams => {
	const params: SearchParams = {
		query: parsed_query.base_query,
	};

	for (const operator of parsed_query.operators) {
		switch (operator.type) {
			case 'site':
				params.include_domains = [
					...(params.include_domains || []),
					operator.value,
				];
				break;
			case 'exclude_site':
				params.exclude_domains = [
					...(params.exclude_domains || []),
					operator.value,
				];
				break;
			case 'filetype':
				params.file_type = operator.value;
				break;
			case 'intitle':
				params.title_filter = operator.value;
				break;
			case 'inurl':
				params.url_filter = operator.value;
				break;
			case 'before':
				params.date_before = operator.value;
				break;
			case 'after':
				params.date_after = operator.value;
				break;
			case 'exact':
				params.exact_phrases = [
					...(params.exact_phrases || []),
					operator.value,
				];
				break;
			case 'boolean':
				// Handle boolean operators in the query string
				if (!params.boolean_operators) {
					params.boolean_operators = [];
				}
				params.boolean_operators.push({
					type: operator.value as 'AND' | 'OR' | 'NOT',
					terms: [],
				});
				break;
		}
	}

	return params;
};
