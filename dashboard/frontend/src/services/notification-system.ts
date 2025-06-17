/**
 * Advanced Notification and Alert Delivery System
 *
 * Provides comprehensive notification capabilities with:
 * - Multiple delivery channels (browser, email, webhook, SMS, push)
 * - Priority-based routing and escalation
 * - Template-based messaging with personalization
 * - Rate limiting and throttling
 * - Delivery tracking and analytics
 * - Offline message queuing
 * - Rich formatting and media support
 * - User preference management
 * - Group and broadcast messaging
 */

export interface NotificationChannel {
  id: string
  type: 'browser' | 'email' | 'webhook' | 'sms' | 'push' | 'slack' | 'teams' | 'discord'
  name: string
  enabled: boolean
  config: any
  priority: number
  rateLimit?: {
    maxMessages: number
    windowMs: number
  }
  retryConfig?: {
    maxRetries: number
    backoffMs: number
    backoffMultiplier: number
  }
}

export interface NotificationTemplate {
  id: string
  name: string
  type: string
  subject?: string
  body: string
  htmlBody?: string
  variables: string[]
  channels: string[]
  priority: 'low' | 'normal' | 'high' | 'critical'
  tags: string[]
  createdAt: Date
  updatedAt: Date
}

export interface NotificationMessage {
  id: string
  templateId?: string
  type: string
  title: string
  body: string
  htmlBody?: string
  priority: 'low' | 'normal' | 'high' | 'critical'
  channels: string[]
  recipients: string[]
  variables?: Record<string, any>
  media?: {
    type: 'image' | 'video' | 'audio' | 'file'
    url: string
    alt?: string
  }[]
  actions?: {
    id: string
    label: string
    url?: string
    action?: string
    style?: 'primary' | 'secondary' | 'danger'
  }[]
  metadata?: Record<string, any>
  scheduledAt?: Date
  expiresAt?: Date
  createdAt: Date
  updatedAt: Date
}

export interface DeliveryAttempt {
  id: string
  messageId: string
  channelId: string
  recipient: string
  status: 'pending' | 'sending' | 'delivered' | 'failed' | 'bounced' | 'expired'
  attemptCount: number
  lastAttemptAt?: Date
  deliveredAt?: Date
  error?: string
  response?: any
  metadata?: Record<string, any>
}

export interface NotificationPreferences {
  userId: string
  channels: {
    [channelType: string]: {
      enabled: boolean
      preferences?: Record<string, any>
    }
  }
  messageTypes: {
    [messageType: string]: {
      enabled: boolean
      channels: string[]
      priority: 'low' | 'normal' | 'high' | 'critical'
    }
  }
  quietHours?: {
    enabled: boolean
    start: string // HH:MM format
    end: string
    timezone: string
    allowCritical: boolean
  }
  rateLimits?: {
    [channelType: string]: {
      maxPerHour: number
      maxPerDay: number
    }
  }
}

export interface NotificationAnalytics {
  messageId: string
  channelStats: {
    [channelId: string]: {
      sent: number
      delivered: number
      failed: number
      bounced: number
      opened?: number
      clicked?: number
      deliveryRate: number
      averageDeliveryTime: number
    }
  }
  recipientStats: {
    [recipient: string]: {
      status: string
      deliveredAt?: Date
      openedAt?: Date
      clickedAt?: Date
      actions: {
        actionId: string
        clickedAt: Date
      }[]
    }
  }
  totalStats: {
    sent: number
    delivered: number
    failed: number
    deliveryRate: number
    averageDeliveryTime: number
  }
}

export class NotificationSystem {
  private channels = new Map<string, NotificationChannel>()
  private templates = new Map<string, NotificationTemplate>()
  private messages = new Map<string, NotificationMessage>()
  private deliveryAttempts = new Map<string, DeliveryAttempt>()
  private userPreferences = new Map<string, NotificationPreferences>()
  private analytics = new Map<string, NotificationAnalytics>()
  private rateLimiters = new Map<string, Map<string, number[]>>()
  private messageQueue: NotificationMessage[] = []
  private processingQueue = false
  private retryQueue: DeliveryAttempt[] = []
  private eventListeners = new Map<string, Set<Function>>()
  private apiBaseUrl: string
  private apiKey: string
  private isOnline = navigator.onLine

  constructor(apiBaseUrl: string, apiKey: string) {
    this.apiBaseUrl = apiBaseUrl
    this.apiKey = apiKey
    this.setupOnlineHandlers()
    this.startQueueProcessor()
    this.startRetryProcessor()
    this.setupBrowserNotifications()
  }

  /**
   * Register a notification channel
   */
  public registerChannel(channel: NotificationChannel): void {
    this.channels.set(channel.id, channel)
    this.emit('channelRegistered', { channel })
  }

  /**
   * Register multiple channels
   */
  public registerChannels(channels: NotificationChannel[]): void {
    channels.forEach((channel) => this.registerChannel(channel))
  }

  /**
   * Create a notification template
   */
  public createTemplate(
    template: Omit<NotificationTemplate, 'id' | 'createdAt' | 'updatedAt'>
  ): NotificationTemplate {
    const fullTemplate: NotificationTemplate = {
      id: this.generateId('tpl'),
      createdAt: new Date(),
      updatedAt: new Date(),
      ...template,
    }

    this.templates.set(fullTemplate.id, fullTemplate)
    this.emit('templateCreated', { template: fullTemplate })
    return fullTemplate
  }

  /**
   * Send a notification message
   */
  public async sendNotification(
    message: Omit<NotificationMessage, 'id' | 'createdAt' | 'updatedAt'>,
    options: {
      immediate?: boolean
      dryRun?: boolean
      trackAnalytics?: boolean
    } = {}
  ): Promise<NotificationMessage> {
    const fullMessage: NotificationMessage = {
      id: this.generateId('msg'),
      createdAt: new Date(),
      updatedAt: new Date(),
      ...message,
    }

    // Process template if specified
    if (fullMessage.templateId) {
      const template = this.templates.get(fullMessage.templateId)
      if (template) {
        fullMessage.body = this.processTemplate(template.body, fullMessage.variables || {})
        if (template.htmlBody) {
          fullMessage.htmlBody = this.processTemplate(
            template.htmlBody,
            fullMessage.variables || {}
          )
        }
        if (template.subject && !fullMessage.title) {
          fullMessage.title = this.processTemplate(template.subject, fullMessage.variables || {})
        }
      }
    }

    this.messages.set(fullMessage.id, fullMessage)

    // Initialize analytics if enabled
    if (options.trackAnalytics !== false) {
      this.initializeAnalytics(fullMessage)
    }

    this.emit('messageCreated', { message: fullMessage })

    // Schedule or queue for immediate sending
    if (fullMessage.scheduledAt && fullMessage.scheduledAt > new Date()) {
      this.scheduleMessage(fullMessage)
    } else if (options.immediate || fullMessage.priority === 'critical') {
      await this.processMessage(fullMessage, options.dryRun)
    } else {
      this.queueMessage(fullMessage)
    }

    return fullMessage
  }

  /**
   * Send a quick notification
   */
  public async sendQuickNotification(
    type: string,
    title: string,
    body: string,
    priority: 'low' | 'normal' | 'high' | 'critical' = 'normal',
    channels?: string[]
  ): Promise<NotificationMessage> {
    return this.sendNotification({
      type,
      title,
      body,
      priority,
      channels: channels || this.getDefaultChannelsForPriority(priority),
      recipients: ['default'],
    })
  }

  /**
   * Send a trading alert
   */
  public async sendTradingAlert(
    alertType:
      | 'trade_executed'
      | 'position_opened'
      | 'position_closed'
      | 'stop_loss_triggered'
      | 'take_profit_hit'
      | 'risk_warning'
      | 'system_error',
    data: Record<string, any>,
    priority: 'low' | 'normal' | 'high' | 'critical' = 'normal'
  ): Promise<NotificationMessage> {
    const template = this.getTradingAlertTemplate(alertType)

    return this.sendNotification({
      templateId: template?.id,
      type: `trading_alert_${alertType}`,
      title: this.formatTradingAlertTitle(alertType, data),
      body: this.formatTradingAlertBody(alertType, data),
      priority,
      channels: this.getChannelsForTradingAlert(alertType, priority),
      recipients: ['trader'],
      variables: data,
      metadata: {
        alertType,
        symbol: data.symbol,
        amount: data.amount,
        timestamp: new Date().toISOString(),
      },
    })
  }

  /**
   * Set user notification preferences
   */
  public setUserPreferences(userId: string, preferences: NotificationPreferences): void {
    this.userPreferences.set(userId, preferences)
    this.emit('preferencesUpdated', { userId, preferences })
  }

  /**
   * Get user notification preferences
   */
  public getUserPreferences(userId: string): NotificationPreferences | undefined {
    return this.userPreferences.get(userId)
  }

  /**
   * Get message analytics
   */
  public getMessageAnalytics(messageId: string): NotificationAnalytics | undefined {
    return this.analytics.get(messageId)
  }

  /**
   * Get delivery attempts for a message
   */
  public getDeliveryAttempts(messageId: string): DeliveryAttempt[] {
    return Array.from(this.deliveryAttempts.values()).filter(
      (attempt) => attempt.messageId === messageId
    )
  }

  /**
   * Test a notification channel
   */
  public async testChannel(channelId: string, testMessage?: string): Promise<boolean> {
    const channel = this.channels.get(channelId)
    if (!channel) {
      throw new Error(`Channel not found: ${channelId}`)
    }

    const testMsg: NotificationMessage = {
      id: this.generateId('test'),
      type: 'test',
      title: 'Test Notification',
      body: testMessage || 'This is a test notification from the trading bot dashboard.',
      priority: 'normal',
      channels: [channelId],
      recipients: ['test'],
      createdAt: new Date(),
      updatedAt: new Date(),
    }

    try {
      await this.deliverToChannel(testMsg, channel, 'test')
      return true
    } catch (error) {
      this.emit('channelTestFailed', { channelId, error })
      return false
    }
  }

  /**
   * Process a notification message
   */
  private async processMessage(message: NotificationMessage, dryRun = false): Promise<void> {
    this.emit('messageProcessing', { message })

    // Check if message has expired
    if (message.expiresAt && message.expiresAt < new Date()) {
      this.emit('messageExpired', { message })
      return
    }

    // Get recipients and apply user preferences
    const recipients = this.resolveRecipients(message)

    // Deliver to each channel
    for (const channelId of message.channels) {
      const channel = this.channels.get(channelId)
      if (!channel?.enabled) {
        continue
      }

      for (const recipient of recipients) {
        // Check user preferences and quiet hours
        if (!this.shouldDeliverToRecipient(message, channel, recipient)) {
          continue
        }

        // Check rate limits
        if (!this.checkRateLimit(channel, recipient)) {
          this.emit('rateLimitExceeded', { channelId, recipient, message })
          continue
        }

        if (!dryRun) {
          await this.createDeliveryAttempt(message, channel, recipient)
        }
      }
    }

    this.emit('messageProcessed', { message, dryRun })
  }

  /**
   * Create a delivery attempt
   */
  private async createDeliveryAttempt(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<void> {
    const attempt: DeliveryAttempt = {
      id: this.generateId('del'),
      messageId: message.id,
      channelId: channel.id,
      recipient,
      status: 'pending',
      attemptCount: 0,
    }

    this.deliveryAttempts.set(attempt.id, attempt)

    try {
      await this.deliverToChannel(message, channel, recipient, attempt)
    } catch (error) {
      attempt.status = 'failed'
      attempt.error = (error as Error).message
      this.queueRetry(attempt)
    }
  }

  /**
   * Deliver message to specific channel
   */
  private async deliverToChannel(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string,
    attempt?: DeliveryAttempt
  ): Promise<void> {
    if (attempt) {
      attempt.status = 'sending'
      attempt.attemptCount++
      attempt.lastAttemptAt = new Date()
    }

    try {
      let result: any

      switch (channel.type) {
        case 'browser':
          result = await this.deliverBrowserNotification(message, channel, recipient)
          break
        case 'email':
          result = await this.deliverEmailNotification(message, channel, recipient)
          break
        case 'webhook':
          result = await this.deliverWebhookNotification(message, channel, recipient)
          break
        case 'sms':
          result = await this.deliverSMSNotification(message, channel, recipient)
          break
        case 'push':
          result = await this.deliverPushNotification(message, channel, recipient)
          break
        case 'slack':
          result = await this.deliverSlackNotification(message, channel, recipient)
          break
        default:
          throw new Error(`Unsupported channel type: ${channel.type}`)
      }

      if (attempt) {
        attempt.status = 'delivered'
        attempt.deliveredAt = new Date()
        attempt.response = result
      }

      this.updateAnalytics(message.id, channel.id, recipient, 'delivered')
      this.emit('messageDelivered', { message, channel, recipient, result })
    } catch (error) {
      if (attempt) {
        attempt.status = 'failed'
        attempt.error = (error as Error).message
      }

      this.updateAnalytics(message.id, channel.id, recipient, 'failed')
      this.emit('deliveryFailed', { message, channel, recipient, error })
      throw error
    }
  }

  /**
   * Deliver browser notification
   */
  private async deliverBrowserNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    if (!('Notification' in window)) {
      throw new Error('Browser notifications not supported')
    }

    if (Notification.permission !== 'granted') {
      throw new Error('Browser notification permission not granted')
    }

    const options: NotificationOptions = {
      body: message.body,
      icon: channel.config?.icon || '/favicon.ico',
      badge: channel.config?.badge,
      image: message.media?.find((m) => m.type === 'image')?.url,
      tag: message.id,
      renotify: message.priority === 'critical',
      requireInteraction: message.priority === 'critical',
      data: {
        messageId: message.id,
        actions: message.actions,
      },
    }

    if (message.actions) {
      options.actions = message.actions.map((action) => ({
        action: action.id,
        title: action.label,
        icon: action.style === 'danger' ? '⚠️' : '✅',
      }))
    }

    const notification = new Notification(message.title, options)

    notification.onclick = () => {
      this.handleNotificationClick(message.id, 'click')
      notification.close()
    }

    notification.onclose = () => {
      this.handleNotificationClose(message.id)
    }

    return { notificationId: message.id, timestamp: new Date() }
  }

  /**
   * Deliver email notification
   */
  private async deliverEmailNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    const emailData = {
      to: recipient,
      subject: message.title,
      text: message.body,
      html: message.htmlBody || this.generateHTMLFromText(message.body),
      priority: message.priority,
      messageId: message.id,
      attachments: message.media
        ?.filter((m) => m.type === 'file')
        .map((m) => ({
          url: m.url,
          filename: m.alt || 'attachment',
        })),
    }

    const response = await fetch(`${this.apiBaseUrl}/api/notifications/email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(emailData),
    })

    if (!response.ok) {
      throw new Error(`Email delivery failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Deliver webhook notification
   */
  private async deliverWebhookNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    const webhookData = {
      message,
      recipient,
      timestamp: new Date().toISOString(),
      channel: channel.id,
    }

    const response = await fetch(channel.config.url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'TradingBot-Dashboard/1.0',
        ...channel.config.headers,
      },
      body: JSON.stringify(webhookData),
    })

    if (!response.ok) {
      throw new Error(`Webhook delivery failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Deliver SMS notification
   */
  private async deliverSMSNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    const smsData = {
      to: recipient,
      body: `${message.title}\n\n${message.body}`,
      messageId: message.id,
    }

    const response = await fetch(`${this.apiBaseUrl}/api/notifications/sms`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(smsData),
    })

    if (!response.ok) {
      throw new Error(`SMS delivery failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Deliver push notification
   */
  private async deliverPushNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    // Implementation for push notifications (Firebase, etc.)
    throw new Error('Push notifications not implemented')
  }

  /**
   * Deliver Slack notification
   */
  private async deliverSlackNotification(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): Promise<any> {
    const slackData = {
      channel: recipient,
      text: message.title,
      blocks: [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: message.title,
          },
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: message.body,
          },
        },
      ],
    }

    if (message.actions) {
      slackData.blocks.push({
        type: 'actions',
        elements: message.actions.map((action) => ({
          type: 'button',
          text: {
            type: 'plain_text',
            text: action.label,
          },
          action_id: action.id,
          url: action.url,
          style: action.style === 'danger' ? 'danger' : 'primary',
        })),
      } as any)
    }

    const response = await fetch(channel.config.webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(slackData),
    })

    if (!response.ok) {
      throw new Error(`Slack delivery failed: ${response.statusText}`)
    }

    return { ok: true }
  }

  /**
   * Utility methods
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private processTemplate(template: string, variables: Record<string, any>): string {
    return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
      return variables[key] !== undefined ? String(variables[key]) : match
    })
  }

  private resolveRecipients(message: NotificationMessage): string[] {
    // Simple resolution - in real implementation, this would resolve groups, roles, etc.
    return message.recipients
  }

  private shouldDeliverToRecipient(
    message: NotificationMessage,
    channel: NotificationChannel,
    recipient: string
  ): boolean {
    const preferences = this.getUserPreferences(recipient)
    if (!preferences) return true

    // Check channel enabled
    const channelPrefs = preferences.channels[channel.type]
    if (channelPrefs && !channelPrefs.enabled) return false

    // Check message type enabled
    const messageTypePrefs = preferences.messageTypes[message.type]
    if (messageTypePrefs && !messageTypePrefs.enabled) return false

    // Check quiet hours
    if (preferences.quietHours?.enabled && message.priority !== 'critical') {
      const now = new Date()
      const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`

      if (
        currentTime >= preferences.quietHours.start &&
        currentTime <= preferences.quietHours.end
      ) {
        if (!preferences.quietHours.allowCritical || (message.priority as string) !== 'critical') {
          return false
        }
      }
    }

    return true
  }

  private checkRateLimit(channel: NotificationChannel, recipient: string): boolean {
    if (!channel.rateLimit) return true

    const key = `${channel.id}_${recipient}`
    const now = Date.now()

    if (!this.rateLimiters.has(channel.id)) {
      this.rateLimiters.set(channel.id, new Map())
    }

    const channelLimiters = this.rateLimiters.get(channel.id)!
    const timestamps = channelLimiters.get(recipient) || []

    // Remove old timestamps
    const cutoff = now - channel.rateLimit.windowMs
    const validTimestamps = timestamps.filter((ts) => ts > cutoff)

    if (validTimestamps.length >= channel.rateLimit.maxMessages) {
      return false
    }

    validTimestamps.push(now)
    channelLimiters.set(recipient, validTimestamps)
    return true
  }

  private getDefaultChannelsForPriority(priority: string): string[] {
    const allChannels = Array.from(this.channels.values())
      .filter((c) => c.enabled)
      .sort((a, b) => b.priority - a.priority)

    switch (priority) {
      case 'critical':
        return allChannels.map((c) => c.id)
      case 'high':
        return allChannels.slice(0, 3).map((c) => c.id)
      case 'normal':
        return allChannels.slice(0, 2).map((c) => c.id)
      default:
        return allChannels.slice(0, 1).map((c) => c.id)
    }
  }

  private getTradingAlertTemplate(alertType: string): NotificationTemplate | undefined {
    return Array.from(this.templates.values()).find((t) => t.type === `trading_alert_${alertType}`)
  }

  private formatTradingAlertTitle(alertType: string, data: Record<string, any>): string {
    const titles = {
      trade_executed: `Trade Executed: ${data.side?.toUpperCase()} ${data.symbol}`,
      position_opened: `Position Opened: ${data.side?.toUpperCase()} ${data.symbol}`,
      position_closed: `Position Closed: ${data.symbol} - ${data.pnl >= 0 ? 'Profit' : 'Loss'}`,
      stop_loss_triggered: `🛑 Stop Loss Triggered: ${data.symbol}`,
      take_profit_hit: `🎯 Take Profit Hit: ${data.symbol}`,
      risk_warning: `⚠️ Risk Warning: ${data.warning}`,
      system_error: `🚨 System Error: ${data.error}`,
    }

    return titles[alertType as keyof typeof titles] || `Trading Alert: ${alertType}`
  }

  private formatTradingAlertBody(alertType: string, data: Record<string, any>): string {
    const bodies = {
      trade_executed: `${data.side?.toUpperCase()} ${data.quantity} ${data.symbol} at $${data.price}\nP&L: $${data.pnl?.toFixed(2) || '0.00'}`,
      position_opened: `Opened ${data.side?.toUpperCase()} position for ${data.symbol}\nSize: ${data.quantity}\nEntry: $${data.entryPrice}`,
      position_closed: `Closed position for ${data.symbol}\nP&L: $${data.pnl?.toFixed(2)}\nDuration: ${data.duration || 'Unknown'}`,
      stop_loss_triggered: `Stop loss triggered for ${data.symbol}\nLoss: $${Math.abs(data.pnl || 0).toFixed(2)}`,
      take_profit_hit: `Take profit target reached for ${data.symbol}\nProfit: $${data.pnl?.toFixed(2)}`,
      risk_warning: data.message || 'Risk threshold exceeded',
      system_error: data.message || 'System error occurred',
    }

    return bodies[alertType as keyof typeof bodies] || `Alert: ${alertType}`
  }

  private getChannelsForTradingAlert(alertType: string, priority: string): string[] {
    // Critical alerts go to all channels
    if (priority === 'critical' || alertType === 'system_error') {
      return Array.from(this.channels.keys())
    }

    // High priority alerts go to primary channels
    if (priority === 'high' || alertType.includes('stop_loss')) {
      return Array.from(this.channels.values())
        .filter((c) => c.enabled && c.priority >= 3)
        .map((c) => c.id)
    }

    // Normal alerts go to browser and email
    return Array.from(this.channels.values())
      .filter((c) => c.enabled && ['browser', 'email'].includes(c.type))
      .map((c) => c.id)
  }

  private initializeAnalytics(message: NotificationMessage): void {
    const analytics: NotificationAnalytics = {
      messageId: message.id,
      channelStats: {},
      recipientStats: {},
      totalStats: {
        sent: 0,
        delivered: 0,
        failed: 0,
        deliveryRate: 0,
        averageDeliveryTime: 0,
      },
    }

    this.analytics.set(message.id, analytics)
  }

  private updateAnalytics(
    messageId: string,
    channelId: string,
    recipient: string,
    status: 'sent' | 'delivered' | 'failed' | 'opened' | 'clicked'
  ): void {
    const analytics = this.analytics.get(messageId)
    if (!analytics) return

    // Update channel stats
    if (!analytics.channelStats[channelId]) {
      analytics.channelStats[channelId] = {
        sent: 0,
        delivered: 0,
        failed: 0,
        bounced: 0,
        deliveryRate: 0,
        averageDeliveryTime: 0,
      }
    }

    const channelStats = analytics.channelStats[channelId]
    if (status === 'sent') channelStats.sent++
    if (status === 'delivered') channelStats.delivered++
    if (status === 'failed') channelStats.failed++

    channelStats.deliveryRate =
      channelStats.sent > 0 ? (channelStats.delivered / channelStats.sent) * 100 : 0

    // Update recipient stats
    if (!analytics.recipientStats[recipient]) {
      analytics.recipientStats[recipient] = {
        status: 'pending',
        actions: [],
      }
    }

    const recipientStats = analytics.recipientStats[recipient]
    if (status === 'delivered') {
      recipientStats.status = 'delivered'
      recipientStats.deliveredAt = new Date()
    } else if (status === 'failed') {
      recipientStats.status = 'failed'
    } else if (status === 'opened') {
      recipientStats.openedAt = new Date()
    } else if (status === 'clicked') {
      recipientStats.clickedAt = new Date()
    }

    // Update total stats
    this.recalculateTotalStats(analytics)
  }

  private recalculateTotalStats(analytics: NotificationAnalytics): void {
    const totals = { sent: 0, delivered: 0, failed: 0 }

    Object.values(analytics.channelStats).forEach((stats) => {
      totals.sent += stats.sent
      totals.delivered += stats.delivered
      totals.failed += stats.failed
    })

    analytics.totalStats = {
      ...totals,
      deliveryRate: totals.sent > 0 ? (totals.delivered / totals.sent) * 100 : 0,
      averageDeliveryTime: 0, // Would be calculated from actual delivery times
    }
  }

  private queueMessage(message: NotificationMessage): void {
    this.messageQueue.push(message)
    this.processQueue()
  }

  private scheduleMessage(message: NotificationMessage): void {
    if (!message.scheduledAt) return

    const delay = message.scheduledAt.getTime() - Date.now()
    if (delay > 0) {
      setTimeout(() => {
        this.processMessage(message)
      }, delay)
    }
  }

  private queueRetry(attempt: DeliveryAttempt): void {
    const channel = this.channels.get(attempt.channelId)
    if (!channel?.retryConfig) return

    const maxRetries = channel.retryConfig.maxRetries
    if (attempt.attemptCount >= maxRetries) {
      this.emit('maxRetriesReached', { attempt })
      return
    }

    const delay =
      channel.retryConfig.backoffMs *
      Math.pow(channel.retryConfig.backoffMultiplier, attempt.attemptCount - 1)

    setTimeout(() => {
      this.retryQueue.push(attempt)
    }, delay)
  }

  private async processQueue(): Promise<void> {
    if (this.processingQueue || this.messageQueue.length === 0) return

    this.processingQueue = true

    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()!
      try {
        await this.processMessage(message)
      } catch (error) {
        this.emit('queueProcessingError', { message, error })
      }

      // Add small delay to prevent overwhelming
      await new Promise((resolve) => setTimeout(resolve, 10))
    }

    this.processingQueue = false
  }

  private startQueueProcessor(): void {
    setInterval(() => {
      this.processQueue()
    }, 1000)
  }

  private startRetryProcessor(): void {
    setInterval(() => {
      this.processRetryQueue()
    }, 5000)
  }

  private async processRetryQueue(): Promise<void> {
    const attempts = this.retryQueue.splice(0)

    for (const attempt of attempts) {
      try {
        const message = this.messages.get(attempt.messageId)
        const channel = this.channels.get(attempt.channelId)

        if (message && channel) {
          await this.deliverToChannel(message, channel, attempt.recipient, attempt)
        }
      } catch (error) {
        this.queueRetry(attempt)
      }
    }
  }

  private setupOnlineHandlers(): void {
    window.addEventListener('online', () => {
      this.isOnline = true
      this.processQueue()
      this.emit('online')
    })

    window.addEventListener('offline', () => {
      this.isOnline = false
      this.emit('offline')
    })
  }

  private async setupBrowserNotifications(): Promise<void> {
    if ('Notification' in window && Notification.permission === 'default') {
      await Notification.requestPermission()
    }
  }

  private handleNotificationClick(messageId: string, action: string): void {
    this.updateAnalytics(messageId, 'browser', 'default', 'clicked')
    this.emit('notificationClicked', { messageId, action })
  }

  private handleNotificationClose(messageId: string): void {
    this.emit('notificationClosed', { messageId })
  }

  private generateHTMLFromText(text: string): string {
    return `<html><body><p>${text.replace(/\n/g, '<br>')}</p></body></html>`
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach((callback) => {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in notification event listener for ${event}:`, error)
        }
      })
    }
  }

  /**
   * Add event listener
   */
  public addEventListener(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  /**
   * Remove event listener
   */
  public removeEventListener(event: string, callback: Function): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventListeners.delete(event)
      }
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    this.messageQueue = []
    this.retryQueue = []
    this.eventListeners.clear()
    this.channels.clear()
    this.templates.clear()
    this.messages.clear()
    this.deliveryAttempts.clear()
    this.analytics.clear()
    this.rateLimiters.clear()
  }
}
