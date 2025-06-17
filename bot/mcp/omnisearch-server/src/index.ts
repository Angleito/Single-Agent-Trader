#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { validate_config } from './config/env.js';
import { initialize_providers } from './providers/index.js';
import { setup_handlers } from './server/handlers.js';
import { register_tools } from './server/tools.js';

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const pkg = JSON.parse(
	readFileSync(join(__dirname, '..', 'package.json'), 'utf8'),
);
const { name, version } = pkg;

class OmnisearchServer {
	private server: Server;

	constructor() {
		this.server = new Server(
			{
				name,
				version,
			},
			{
				capabilities: {
					tools: {},
					resources: {},
				},
			},
		);

		// Validate environment configuration
		validate_config();

		// Initialize and register providers
		initialize_providers();

		// Register tools and setup handlers
		register_tools(this.server);
		setup_handlers(this.server);

		// Error handling
		this.server.onerror = (error: Error) =>
			console.error('[MCP Error]', error);
		process.on('SIGINT', async () => {
			await this.server.close();
			process.exit(0);
		});
	}

	async run() {
		const transport = new StdioServerTransport();
		await this.server.connect(transport);
		console.error('Omnisearch MCP server running on stdio');
	}
}

const server = new OmnisearchServer();
server.run().catch(console.error);
