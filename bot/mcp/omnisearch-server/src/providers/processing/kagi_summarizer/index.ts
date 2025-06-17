import {
	ErrorType,
	ProcessingProvider,
	ProcessingResult,
	ProviderError,
} from '../../../common/types.js';
import {
	handle_rate_limit,
	retry_with_backoff,
	validate_api_key,
} from '../../../common/utils.js';
import { config } from '../../../config/env.js';

interface KagiSummarizerResponse {
	meta: {
		id: string;
		node: string;
		ms: number;
		api_balance: number;
	};
	data: {
		output: string;
		tokens: number;
	};
}

export class KagiSummarizerProvider implements ProcessingProvider {
	name = 'kagi_summarizer';
	description =
		'Instantly summarizes content of any type and length from URLs. Supports pages, videos, and podcasts with transcripts. Best for quick comprehension of long-form content and multimedia resources.';

	async process_content(url: string): Promise<ProcessingResult> {
		const api_key = validate_api_key(
			config.processing.kagi_summarizer.api_key,
			this.name,
		);

		const summarize_request = async () => {
			try {
				const response = await fetch(
					config.processing.kagi_summarizer.base_url,
					{
						method: 'POST',
						headers: {
							'Content-Type': 'application/json',
							Authorization: `Bot ${api_key}`,
						},
						body: JSON.stringify({ url }),
						signal: AbortSignal.timeout(
							config.processing.kagi_summarizer.timeout,
						),
					},
				);

				let data: KagiSummarizerResponse & { message?: string };
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

				if (!response.ok || !data.data?.output) {
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
								'Kagi Summarizer API internal error',
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

				return {
					content: data.data.output,
					metadata: {
						word_count: data.data.tokens,
					},
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

		return retry_with_backoff(summarize_request);
	}
}
