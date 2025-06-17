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

export class JinaReaderProvider implements ProcessingProvider {
	name = 'jina_reader';
	description =
		'Convert any URL to clean, LLM-friendly text using Jina Reader API';

	constructor() {
		// Validate API key exists at construction time
		validate_api_key(
			config.processing.jina_reader.api_key,
			this.name,
		);
	}

	async process_content(url: string): Promise<ProcessingResult> {
		if (!is_valid_url(url)) {
			throw new ProviderError(
				ErrorType.INVALID_INPUT,
				'Invalid URL provided',
				this.name,
			);
		}

		const process_url = async () => {
			const api_key = validate_api_key(
				config.processing.jina_reader.api_key,
				this.name,
			);

			const headers: HeadersInit = {
				Authorization: `Bearer ${api_key}`,
				Accept: 'application/json',
				'Content-Type': 'application/json',
			};

			const response = await fetch('https://r.jina.ai/', {
				method: 'POST',
				headers,
				body: JSON.stringify({ url }),
			});

			if (!response.ok) {
				if (response.status === 429) {
					throw new ProviderError(
						ErrorType.RATE_LIMIT,
						'Rate limit exceeded',
						this.name,
					);
				}

				throw new ProviderError(
					ErrorType.API_ERROR,
					`API request failed with status ${response.status}`,
					this.name,
				);
			}

			const data = await response.json();

			if (!data.data) {
				throw new ProviderError(
					ErrorType.API_ERROR,
					'Invalid response format from Jina Reader',
					this.name,
				);
			}

			return {
				content: data.data.content || '',
				metadata: {
					title: data.data.title || '',
					date: data.data.timestamp || '',
					word_count: (data.data.content || '')
						.split(/\s+/)
						.filter(Boolean).length,
				},
				source_provider: this.name,
			};
		};

		try {
			return await retry_with_backoff(process_url);
		} catch (error: unknown) {
			if (error instanceof ProviderError) {
				throw error;
			}
			throw new ProviderError(
				ErrorType.PROVIDER_ERROR,
				`Failed to process content: ${
					error instanceof Error ? error.message : String(error)
				}`,
				this.name,
			);
		}
	}
}
