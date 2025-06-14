# Spot to Futures Transfer Guide

This guide explains how to transfer funds from your Coinbase spot account (CBI) to your futures account (CFM).

## Prerequisites

1. **Environment Variables**: Make sure you have your CDP credentials set in `.env`:
   ```
   EXCHANGE__CDP_API_KEY_NAME=your_cdp_key_name
   EXCHANGE__CDP_PRIVATE_KEY=your_cdp_private_key
   ```

2. **Account Access**: Your CDP API key must have permissions to:
   - Read account balances
   - Execute transfers between accounts

3. **Futures Account**: You must have a Coinbase futures account enabled

## Using the Transfer Script

### Interactive Mode
Run the script without arguments to enter the amount interactively:
```bash
python transfer_to_futures.py
```

The script will:
1. Show your current spot and futures balances
2. Prompt you for the transfer amount
3. Show a summary and ask for confirmation
4. Execute the transfer
5. Display updated balances

### Command Line Mode
Specify the amount directly:
```bash
python transfer_to_futures.py 1000
```
This will transfer $1000 from spot to futures.

### Example Output
```
Initializing Coinbase client...
Connecting to Coinbase...
Successfully connected to Coinbase

Fetching current balances...

Current Balances:
  Spot (CBI):    $5,000.00
  Futures (CFM): $2,000.00
  Total:         $7,000.00

Available for transfer: $5,000.00
Enter amount to transfer to futures (USD): $1000

Transfer Summary:
  From: Spot (CBI)
  To:   Futures (CFM)
  Amount: $1,000.00

New balances after transfer:
  Spot:    $4,000.00
  Futures: $3,000.00

Proceed with transfer? (y/n): y

Executing transfer of $1,000.00 to futures...
✅ Transfer successful!

Fetching updated balances...

Updated Balances:
  Spot (CBI):    $4,000.00
  Futures (CFM): $3,000.00
  Total:         $7,000.00

Disconnected from Coinbase
```

## Important Notes

1. **Minimum Transfer**: Coinbase may have minimum transfer amounts (typically $100)

2. **Transfer Time**: Transfers are usually instant but may take a few seconds to reflect

3. **Auto Transfer**: The trading bot can automatically transfer funds when needed if `TRADING__AUTO_CASH_TRANSFER=true`

4. **Dry Run**: The transfer script always runs in live mode. Make sure you want to transfer real funds.

## Troubleshooting

### Transfer Failed
If the transfer fails, check:
- Your API credentials have transfer permissions
- You have sufficient balance in spot account
- The amount meets minimum requirements
- Auto cash transfer is enabled in the bot config

### Connection Failed
If connection fails:
- Verify your CDP credentials in `.env`
- Check if you're using the correct API key format
- Ensure your private key is in PEM format

### API Errors
Common API errors:
- "Insufficient funds": Not enough balance in spot account
- "Invalid amount": Amount might be below minimum or have too many decimals
- "Permission denied": API key lacks transfer permissions

## Alternative Methods

### Using the Coinbase Web Interface
1. Log into Coinbase Advanced
2. Go to Portfolio → Transfers
3. Select "Transfer between portfolios"
4. Choose CBI → CFM transfer

### Using the Trading Bot
The bot can automatically transfer funds when opening futures positions if:
- `TRADING__ENABLE_FUTURES=true`
- `TRADING__AUTO_CASH_TRANSFER=true`
- Sufficient spot balance is available