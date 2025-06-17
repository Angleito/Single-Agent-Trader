import {
	EnhancementProvider,
	EnhancementResult,
	ErrorType,
	ProviderError,
} from '../../../common/types.js';
import {
	handle_rate_limit,
	retry_with_backoff,
	validate_api_key,
} from '../../../common/utils.js';
import { config } from '../../../config/env.js';

interface JinaGroundingResponse {
	code: number;
	status: number;
	data: {
		factuality: number;
		result: boolean;
		reason: string;
		references: Array<{
			url: string;
			key_quote: string;
			is_supportive: boolean;
		}>;
		usage: {
			tokens: number;
		};
	};
}

export class JinaGroundingProvider implements EnhancementProvider {
	name = 'jina_grounding';
	description =
		'Real-time fact verification against web knowledge. Reduces hallucinations and improves content integrity through statement verification.';

	async enhance_content(content: string): Promise<EnhancementResult> {
		const api_key = validate_api_key(
			config.enhancement.jina_grounding.api_key,
			this.name,
		);

		const ground_request = async () => {
			try {
				const response = await fetch('https://g.jina.ai', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${api_key}`,
					},
					body: JSON.stringify({ statement: content }),
					signal: AbortSignal.timeout(
						config.enhancement.jina_grounding.timeout,
					),
				});

				let data: JinaGroundingResponse;
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
								'Jina Grounding API internal error',
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

				// Format references into a readable string
				const references_text = data.data.references
					.map(
						(ref) =>
							`${ref.is_supportive ? '✓' : '✗'} ${ref.key_quote} (${
								ref.url
							})`,
					)
					.join('\n\n');

				return {
					original_content: content,
					enhanced_content: `Factuality Score: ${
						data.data.factuality
					}\nVerdict: ${
						data.data.result ? 'True' : 'False'
					}\n\nReasoning: ${
						data.data.reason
					}\n\nReferences:\n${references_text}`,
					enhancements: [
						{
							type: 'fact_verification',
							description:
								'Verified factual accuracy against real-time web knowledge',
						},
					],
					sources: data.data.references.map((ref) => ({
						title: ref.key_quote,
						url: ref.url,
					})),
					source_provider: this.name,
					meta: {
						factuality: data.data.factuality,
						result: data.data.result,
						token_usage: data.data.usage.tokens,
					},
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

		return retry_with_backoff(ground_request);
	}
}
