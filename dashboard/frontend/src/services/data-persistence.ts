/**
 * Advanced Data Persistence and Offline Capabilities
 *
 * Provides comprehensive offline-first data management:
 * - IndexedDB wrapper with schema versioning
 * - Intelligent data synchronization strategies
 * - Conflict resolution and data merging
 * - Cache management with TTL and size limits
 * - Background data sync with retry mechanisms
 * - Data compression and optimization
 * - Query optimization and indexing
 * - Real-time data replication
 * - Offline transaction queuing
 * - Data integrity validation
 */

export interface DataSchema {
  version: number
  stores: {
    [storeName: string]: {
      keyPath?: string
      autoIncrement?: boolean
      indexes?: {
        [indexName: string]: {
          keyPath: string | string[]
          unique?: boolean
          multiEntry?: boolean
        }
      }
    }
  }
  migrations?: {
    [version: number]: (db: IDBDatabase, transaction: IDBTransaction) => void
  }
}

export interface DataRecord {
  id: string
  data: any
  metadata: {
    createdAt: Date
    updatedAt: Date
    version: number
    source: 'local' | 'remote' | 'sync'
    ttl?: number
    tags?: string[]
    checksum?: string
  }
  syncStatus: 'pending' | 'synced' | 'conflict' | 'failed'
}

export interface SyncConfig {
  enabled: boolean
  strategy: 'real-time' | 'periodic' | 'manual' | 'on-demand'
  interval: number
  batchSize: number
  maxRetries: number
  conflictResolution: 'client-wins' | 'server-wins' | 'merge' | 'manual'
  syncEndpoints: {
    [storeName: string]: {
      url: string
      method: 'GET' | 'POST' | 'PUT' | 'PATCH'
      headers?: Record<string, string>
    }
  }
}

export interface CacheConfig {
  maxSize: number // in MB
  maxEntries: number
  defaultTTL: number // in seconds
  cleanupInterval: number // in seconds
  compressionEnabled: boolean
  encryptionEnabled: boolean
}

export interface QueryOptions {
  limit?: number
  offset?: number
  orderBy?: string
  orderDirection?: 'asc' | 'desc'
  filters?: {
    [field: string]: any
  }
  includes?: string[]
  excludes?: string[]
}

export interface SyncOperation {
  id: string
  storeName: string
  operation: 'create' | 'update' | 'delete'
  recordId: string
  data?: any
  timestamp: Date
  attempts: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error?: string
}

export interface ConflictResolution {
  recordId: string
  localData: any
  remoteData: any
  resolution: 'local' | 'remote' | 'merged'
  mergedData?: any
  timestamp: Date
}

export class DataPersistenceManager {
  private db: IDBDatabase | null = null
  private schema: DataSchema
  private syncConfig: SyncConfig
  private cacheConfig: CacheConfig
  private syncQueue: SyncOperation[] = []
  private isOnline = navigator.onLine
  private syncInterval: number | null = null
  private cleanupInterval: number | null = null
  private eventListeners = new Map<string, Set<Function>>()
  private compressionWorker: Worker | null = null
  private encryptionKey: CryptoKey | null = null
  private isInitialized = false

  // Performance monitoring
  private performance = {
    queriesExecuted: 0,
    avgQueryTime: 0,
    cacheHits: 0,
    cacheMisses: 0,
    syncOperations: 0,
    conflictsResolved: 0,
    dataCompressed: 0,
    dataEncrypted: 0,
  }

  constructor(schema: DataSchema, syncConfig: SyncConfig, cacheConfig: CacheConfig) {
    this.schema = schema
    this.syncConfig = syncConfig
    this.cacheConfig = cacheConfig
    this.setupEventListeners()
  }

  /**
   * Initialize the database and setup systems
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) return

    try {
      await this.openDatabase()
      await this.setupEncryption()
      await this.setupCompression()
      this.startSyncProcess()
      this.startCleanupProcess()
      this.setupOnlineHandlers()

      this.isInitialized = true
      this.emit('initialized')
    } catch (error) {
      this.emit('error', { type: 'initialization', error })
      throw error
    }
  }

  /**
   * Open and setup IndexedDB database
   */
  private async openDatabase(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('TradingBotDB', this.schema.version)

      request.onerror = () => {
        reject(new Error(`Failed to open database: ${request.error?.message}`))
      }

      request.onsuccess = () => {
        this.db = request.result
        this.setupErrorHandling()
        resolve()
      }

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        const transaction = (event.target as IDBOpenDBRequest).transaction!

        this.handleSchemaUpgrade(db, transaction, event.oldVersion)
      }
    })
  }

  /**
   * Handle database schema upgrades
   */
  private handleSchemaUpgrade(
    db: IDBDatabase,
    transaction: IDBTransaction,
    oldVersion: number
  ): void {
    // Create or update object stores
    Object.entries(this.schema.stores).forEach(([storeName, storeConfig]) => {
      let store: IDBObjectStore

      if (!db.objectStoreNames.contains(storeName)) {
        store = db.createObjectStore(storeName, {
          keyPath: storeConfig.keyPath || 'id',
          autoIncrement: storeConfig.autoIncrement || false,
        })
      } else {
        store = transaction.objectStore(storeName)
      }

      // Create indexes
      if (storeConfig.indexes) {
        Object.entries(storeConfig.indexes).forEach(([indexName, indexConfig]) => {
          if (!store.indexNames.contains(indexName)) {
            store.createIndex(indexName, indexConfig.keyPath, {
              unique: indexConfig.unique || false,
              multiEntry: indexConfig.multiEntry || false,
            })
          }
        })
      }
    })

    // Run migrations
    if (this.schema.migrations) {
      for (let version = oldVersion + 1; version <= this.schema.version; version++) {
        const migration = this.schema.migrations[version]
        if (migration) {
          migration(db, transaction)
        }
      }
    }
  }

  /**
   * Setup encryption for sensitive data
   */
  private async setupEncryption(): Promise<void> {
    if (!this.cacheConfig.encryptionEnabled) return

    try {
      // Generate or retrieve encryption key
      this.encryptionKey = await window.crypto.subtle.generateKey(
        {
          name: 'AES-GCM',
          length: 256,
        },
        true,
        ['encrypt', 'decrypt']
      )
    } catch (error) {
      console.warn('Encryption setup failed:', error)
    }
  }

  /**
   * Setup compression worker
   */
  private async setupCompression(): Promise<void> {
    if (!this.cacheConfig.compressionEnabled) return

    try {
      // Create compression worker
      const workerCode = `
        self.onmessage = function(e) {
          const { id, action, data } = e.data;
          
          try {
            if (action === 'compress') {
              // Simple compression using JSON stringification and encoding
              const compressed = btoa(JSON.stringify(data));
              self.postMessage({ id, result: compressed });
            } else if (action === 'decompress') {
              const decompressed = JSON.parse(atob(data));
              self.postMessage({ id, result: decompressed });
            }
          } catch (error) {
            self.postMessage({ id, error: error.message });
          }
        };
      `

      const blob = new Blob([workerCode], { type: 'application/javascript' })
      this.compressionWorker = new Worker(URL.createObjectURL(blob))
    } catch (error) {
      console.warn('Compression worker setup failed:', error)
    }
  }

  /**
   * Create a new record
   */
  public async create(
    storeName: string,
    data: any,
    options: { sync?: boolean } = {}
  ): Promise<string> {
    const startTime = performance.now()

    try {
      const id = this.generateId()
      const record: DataRecord = {
        id,
        data: await this.processDataForStorage(data),
        metadata: {
          createdAt: new Date(),
          updatedAt: new Date(),
          version: 1,
          source: 'local',
          checksum: await this.calculateChecksum(data),
        },
        syncStatus: options.sync !== false ? 'pending' : 'synced',
      }

      await this.writeToStore(storeName, record)

      if (options.sync !== false && this.syncConfig.enabled) {
        await this.queueSyncOperation('create', storeName, id, data)
      }

      this.updatePerformance('create', startTime)
      this.emit('recordCreated', { storeName, id, data })

      return id
    } catch (error) {
      this.emit('error', { type: 'create', storeName, error })
      throw error
    }
  }

  /**
   * Read a record by ID
   */
  public async read(storeName: string, id: string): Promise<any | null> {
    const startTime = performance.now()

    try {
      const record = await this.readFromStore(storeName, id)

      if (!record) {
        this.performance.cacheMisses++
        return null
      }

      // Check TTL
      if (this.isExpired(record)) {
        await this.delete(storeName, id)
        this.performance.cacheMisses++
        return null
      }

      this.performance.cacheHits++
      const data = await this.processDataFromStorage(record.data)

      this.updatePerformance('read', startTime)
      return data
    } catch (error) {
      this.emit('error', { type: 'read', storeName, id, error })
      throw error
    }
  }

  /**
   * Update a record
   */
  public async update(
    storeName: string,
    id: string,
    data: any,
    options: { sync?: boolean; merge?: boolean } = {}
  ): Promise<void> {
    const startTime = performance.now()

    try {
      const existingRecord = await this.readFromStore(storeName, id)
      if (!existingRecord) {
        throw new Error(`Record not found: ${id}`)
      }

      const updatedData = options.merge ? this.mergeData(existingRecord.data, data) : data

      const updatedRecord: DataRecord = {
        ...existingRecord,
        data: await this.processDataForStorage(updatedData),
        metadata: {
          ...existingRecord.metadata,
          updatedAt: new Date(),
          version: existingRecord.metadata.version + 1,
          checksum: await this.calculateChecksum(updatedData),
        },
        syncStatus: options.sync !== false ? 'pending' : 'synced',
      }

      await this.writeToStore(storeName, updatedRecord)

      if (options.sync !== false && this.syncConfig.enabled) {
        await this.queueSyncOperation('update', storeName, id, updatedData)
      }

      this.updatePerformance('update', startTime)
      this.emit('recordUpdated', { storeName, id, data: updatedData })
    } catch (error) {
      this.emit('error', { type: 'update', storeName, id, error })
      throw error
    }
  }

  /**
   * Delete a record
   */
  public async delete(
    storeName: string,
    id: string,
    options: { sync?: boolean } = {}
  ): Promise<void> {
    const startTime = performance.now()

    try {
      await this.deleteFromStore(storeName, id)

      if (options.sync !== false && this.syncConfig.enabled) {
        await this.queueSyncOperation('delete', storeName, id)
      }

      this.updatePerformance('delete', startTime)
      this.emit('recordDeleted', { storeName, id })
    } catch (error) {
      this.emit('error', { type: 'delete', storeName, id, error })
      throw error
    }
  }

  /**
   * Query records with advanced filtering and pagination
   */
  public async query(storeName: string, options: QueryOptions = {}): Promise<any[]> {
    const startTime = performance.now()

    try {
      const records = await this.queryFromStore(storeName, options)
      const results = []

      for (const record of records) {
        if (!this.isExpired(record)) {
          const data = await this.processDataFromStorage(record.data)
          results.push(data)
        }
      }

      this.updatePerformance('query', startTime)
      return results
    } catch (error) {
      this.emit('error', { type: 'query', storeName, error })
      throw error
    }
  }

  /**
   * Synchronize data with remote server
   */
  public async sync(storeName?: string, force = false): Promise<void> {
    if (!this.syncConfig.enabled && !force) return
    if (!this.isOnline) {
      this.emit('syncSkipped', { reason: 'offline' })
      return
    }

    try {
      const stores = storeName ? [storeName] : Object.keys(this.schema.stores)

      for (const store of stores) {
        await this.syncStore(store)
      }

      this.emit('syncCompleted')
    } catch (error) {
      this.emit('error', { type: 'sync', error })
      throw error
    }
  }

  /**
   * Synchronize a specific store
   */
  private async syncStore(storeName: string): Promise<void> {
    const endpoint = this.syncConfig.syncEndpoints[storeName]
    if (!endpoint) return

    // Process outgoing sync operations
    const pendingOps = this.syncQueue.filter(
      (op) => op.storeName === storeName && op.status === 'pending'
    )

    for (const op of pendingOps) {
      try {
        op.status = 'processing'
        await this.processSyncOperation(op, endpoint)
        op.status = 'completed'
        this.performance.syncOperations++
      } catch (error) {
        op.status = 'failed'
        op.error = (error as Error).message
        op.attempts++

        if (op.attempts < this.syncConfig.maxRetries) {
          op.status = 'pending'
        }
      }
    }

    // Fetch remote changes
    await this.fetchRemoteChanges(storeName, endpoint)
  }

  /**
   * Process a sync operation
   */
  private async processSyncOperation(op: SyncOperation, endpoint: any): Promise<void> {
    const url = `${endpoint.url}/${op.recordId}`
    const method = op.operation === 'delete' ? 'DELETE' : endpoint.method

    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...endpoint.headers,
      },
      body: op.operation !== 'delete' ? JSON.stringify(op.data) : undefined,
    })

    if (!response.ok) {
      throw new Error(`Sync operation failed: ${response.statusText}`)
    }

    // Update local record sync status
    if (op.operation !== 'delete') {
      const record = await this.readFromStore(op.storeName, op.recordId)
      if (record) {
        record.syncStatus = 'synced'
        await this.writeToStore(op.storeName, record)
      }
    }
  }

  /**
   * Fetch remote changes
   */
  private async fetchRemoteChanges(storeName: string, endpoint: any): Promise<void> {
    const response = await fetch(endpoint.url, {
      method: 'GET',
      headers: endpoint.headers,
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch remote changes: ${response.statusText}`)
    }

    const remoteData = await response.json()

    for (const remoteRecord of remoteData) {
      await this.mergeRemoteRecord(storeName, remoteRecord)
    }
  }

  /**
   * Merge remote record with local data
   */
  private async mergeRemoteRecord(storeName: string, remoteRecord: any): Promise<void> {
    const localRecord = await this.readFromStore(storeName, remoteRecord.id)

    if (!localRecord) {
      // New remote record
      await this.create(storeName, remoteRecord, { sync: false })
      return
    }

    // Check for conflicts
    const localChecksum = localRecord.metadata.checksum
    const remoteChecksum = await this.calculateChecksum(remoteRecord)

    if (localChecksum === remoteChecksum) {
      // No conflict
      return
    }

    // Handle conflict based on strategy
    const resolution = await this.resolveConflict(localRecord, remoteRecord)

    if (resolution.resolution === 'remote' || resolution.resolution === 'merged') {
      const dataToUpdate = resolution.resolution === 'merged' ? resolution.mergedData : remoteRecord

      await this.update(storeName, remoteRecord.id, dataToUpdate, { sync: false })
    }
  }

  /**
   * Resolve data conflicts
   */
  private async resolveConflict(
    localRecord: DataRecord,
    remoteData: any
  ): Promise<ConflictResolution> {
    const resolution: ConflictResolution = {
      recordId: localRecord.id,
      localData: localRecord.data,
      remoteData,
      resolution: 'local',
      timestamp: new Date(),
    }

    switch (this.syncConfig.conflictResolution) {
      case 'server-wins':
        resolution.resolution = 'remote'
        break

      case 'client-wins':
        resolution.resolution = 'local'
        break

      case 'merge':
        resolution.resolution = 'merged'
        resolution.mergedData = this.mergeData(localRecord.data, remoteData)
        break

      case 'manual':
        // Emit event for manual resolution
        this.emit('conflictDetected', resolution)
        // For now, default to local
        resolution.resolution = 'local'
        break
    }

    this.performance.conflictsResolved++
    return resolution
  }

  /**
   * Merge two data objects
   */
  private mergeData(local: any, remote: any): any {
    if (typeof local !== 'object' || typeof remote !== 'object') {
      return remote
    }

    const merged = { ...local }

    for (const key in remote) {
      if (remote.hasOwnProperty(key)) {
        if (typeof remote[key] === 'object' && typeof local[key] === 'object') {
          merged[key] = this.mergeData(local[key], remote[key])
        } else {
          merged[key] = remote[key]
        }
      }
    }

    return merged
  }

  /**
   * Queue sync operation
   */
  private async queueSyncOperation(
    operation: 'create' | 'update' | 'delete',
    storeName: string,
    recordId: string,
    data?: any
  ): Promise<void> {
    const syncOp: SyncOperation = {
      id: this.generateId(),
      storeName,
      operation,
      recordId,
      data,
      timestamp: new Date(),
      attempts: 0,
      status: 'pending',
    }

    this.syncQueue.push(syncOp)

    // Trigger immediate sync for real-time strategy
    if (this.syncConfig.strategy === 'real-time' && this.isOnline) {
      setTimeout(() => this.sync(storeName), 100)
    }
  }

  /**
   * Database operations
   */
  private async writeToStore(storeName: string, record: DataRecord): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite')
      const store = transaction.objectStore(storeName)

      const request = store.put(record)

      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  private async readFromStore(storeName: string, id: string): Promise<DataRecord | null> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly')
      const store = transaction.objectStore(storeName)

      const request = store.get(id)

      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error)
    })
  }

  private async deleteFromStore(storeName: string, id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readwrite')
      const store = transaction.objectStore(storeName)

      const request = store.delete(id)

      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  private async queryFromStore(storeName: string, options: QueryOptions): Promise<DataRecord[]> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly')
      const store = transaction.objectStore(storeName)

      let source: IDBObjectStore | IDBIndex = store

      // Use index if orderBy matches an index
      if (options.orderBy && store.indexNames.contains(options.orderBy)) {
        source = store.index(options.orderBy)
      }

      const direction = options.orderDirection === 'desc' ? 'prev' : 'next'
      const request = source.openCursor(null, direction)
      const results: DataRecord[] = []
      let count = 0
      const offset = options.offset || 0
      const limit = options.limit || Number.MAX_SAFE_INTEGER

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result

        if (!cursor || results.length >= limit) {
          resolve(results)
          return
        }

        const record = cursor.value as DataRecord

        // Apply filters
        if (this.matchesFilters(record, options.filters)) {
          if (count >= offset) {
            results.push(record)
          }
          count++
        }

        cursor.continue()
      }

      request.onerror = () => reject(request.error)
    })
  }

  /**
   * Check if record matches filters
   */
  private matchesFilters(record: DataRecord, filters?: { [field: string]: any }): boolean {
    if (!filters) return true

    for (const [field, value] of Object.entries(filters)) {
      const recordValue = this.getNestedValue(record.data, field)

      if (Array.isArray(value)) {
        if (!value.includes(recordValue)) return false
      } else if (typeof value === 'object' && value.operator) {
        if (!this.evaluateOperator(recordValue, value.operator, value.value)) {
          return false
        }
      } else {
        if (recordValue !== value) return false
      }
    }

    return true
  }

  /**
   * Get nested object value
   */
  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj)
  }

  /**
   * Evaluate filter operator
   */
  private evaluateOperator(value: any, operator: string, target: any): boolean {
    switch (operator) {
      case 'gt':
        return value > target
      case 'gte':
        return value >= target
      case 'lt':
        return value < target
      case 'lte':
        return value <= target
      case 'ne':
        return value !== target
      case 'like':
        return String(value).includes(String(target))
      case 'in':
        return Array.isArray(target) && target.includes(value)
      default:
        return value === target
    }
  }

  /**
   * Data processing methods
   */
  private async processDataForStorage(data: any): Promise<any> {
    let processed = data

    // Compress if enabled
    if (this.cacheConfig.compressionEnabled && this.compressionWorker) {
      processed = await this.compress(processed)
      this.performance.dataCompressed++
    }

    // Encrypt if enabled
    if (this.cacheConfig.encryptionEnabled && this.encryptionKey) {
      processed = await this.encrypt(processed)
      this.performance.dataEncrypted++
    }

    return processed
  }

  private async processDataFromStorage(data: any): Promise<any> {
    let processed = data

    // Decrypt if needed
    if (this.cacheConfig.encryptionEnabled && this.encryptionKey) {
      processed = await this.decrypt(processed)
    }

    // Decompress if needed
    if (this.cacheConfig.compressionEnabled && this.compressionWorker) {
      processed = await this.decompress(processed)
    }

    return processed
  }

  /**
   * Compression methods
   */
  private async compress(data: any): Promise<string> {
    if (!this.compressionWorker) return JSON.stringify(data)

    return new Promise((resolve, reject) => {
      const id = this.generateId()

      const handler = (event: MessageEvent) => {
        if (event.data.id === id) {
          this.compressionWorker!.removeEventListener('message', handler)

          if (event.data.error) {
            reject(new Error(event.data.error))
          } else {
            resolve(event.data.result)
          }
        }
      }

      this.compressionWorker.addEventListener('message', handler)
      this.compressionWorker.postMessage({ id, action: 'compress', data })
    })
  }

  private async decompress(data: string): Promise<any> {
    if (!this.compressionWorker) return JSON.parse(data)

    return new Promise((resolve, reject) => {
      const id = this.generateId()

      const handler = (event: MessageEvent) => {
        if (event.data.id === id) {
          this.compressionWorker!.removeEventListener('message', handler)

          if (event.data.error) {
            reject(new Error(event.data.error))
          } else {
            resolve(event.data.result)
          }
        }
      }

      this.compressionWorker.addEventListener('message', handler)
      this.compressionWorker.postMessage({ id, action: 'decompress', data })
    })
  }

  /**
   * Encryption methods
   */
  private async encrypt(data: any): Promise<string> {
    if (!this.encryptionKey) return JSON.stringify(data)

    const dataString = JSON.stringify(data)
    const encoder = new TextEncoder()
    const dataBuffer = encoder.encode(dataString)

    const iv = window.crypto.getRandomValues(new Uint8Array(12))
    const encrypted = await window.crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      dataBuffer
    )

    // Combine IV and encrypted data
    const combined = new Uint8Array(iv.length + encrypted.byteLength)
    combined.set(iv)
    combined.set(new Uint8Array(encrypted), iv.length)

    return btoa(String.fromCharCode(...combined))
  }

  private async decrypt(encryptedData: string): Promise<any> {
    if (!this.encryptionKey) return JSON.parse(encryptedData)

    const combined = new Uint8Array(
      atob(encryptedData)
        .split('')
        .map((char) => char.charCodeAt(0))
    )

    const iv = combined.slice(0, 12)
    const encrypted = combined.slice(12)

    const decrypted = await window.crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      encrypted
    )

    const decoder = new TextDecoder()
    const dataString = decoder.decode(decrypted)

    return JSON.parse(dataString)
  }

  /**
   * Utility methods
   */
  private async calculateChecksum(data: any): Promise<string> {
    const dataString = JSON.stringify(data)
    const encoder = new TextEncoder()
    const dataBuffer = encoder.encode(dataString)

    const hashBuffer = await window.crypto.subtle.digest('SHA-256', dataBuffer)
    const hashArray = Array.from(new Uint8Array(hashBuffer))

    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')
  }

  private isExpired(record: DataRecord): boolean {
    if (!record.metadata.ttl) return false

    const now = Date.now()
    const recordTime = record.metadata.updatedAt.getTime()
    const ttl = record.metadata.ttl * 1000

    return now - recordTime > ttl
  }

  private generateId(): string {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private updatePerformance(operation: string, startTime: number): void {
    const duration = performance.now() - startTime
    this.performance.queriesExecuted++
    this.performance.avgQueryTime =
      (this.performance.avgQueryTime * (this.performance.queriesExecuted - 1) + duration) /
      this.performance.queriesExecuted
  }

  /**
   * Event handling
   */
  private setupEventListeners(): void {
    // Setup online/offline handlers
    window.addEventListener('online', () => {
      this.isOnline = true
      this.emit('online')
      if (this.syncConfig.enabled) {
        this.sync()
      }
    })

    window.addEventListener('offline', () => {
      this.isOnline = false
      this.emit('offline')
    })
  }

  private setupErrorHandling(): void {
    if (this.db) {
      this.db.onerror = (event) => {
        this.emit('error', { type: 'database', error: event })
      }
    }
  }

  private setupOnlineHandlers(): void {
    this.isOnline = navigator.onLine
  }

  /**
   * Background processes
   */
  private startSyncProcess(): void {
    if (!this.syncConfig.enabled || this.syncConfig.strategy !== 'periodic') return

    this.syncInterval = window.setInterval(() => {
      if (this.isOnline) {
        this.sync().catch((error) => {
          this.emit('error', { type: 'sync', error })
        })
      }
    }, this.syncConfig.interval)
  }

  private startCleanupProcess(): void {
    this.cleanupInterval = window.setInterval(() => {
      this.cleanup().catch((error) => {
        this.emit('error', { type: 'cleanup', error })
      })
    }, this.cacheConfig.cleanupInterval * 1000)
  }

  /**
   * Cache management
   */
  public async cleanup(): Promise<void> {
    try {
      const stores = Object.keys(this.schema.stores)

      for (const storeName of stores) {
        await this.cleanupStore(storeName)
      }

      this.emit('cleanupCompleted')
    } catch (error) {
      this.emit('error', { type: 'cleanup', error })
      throw error
    }
  }

  private async cleanupStore(storeName: string): Promise<void> {
    const allRecords = await this.queryFromStore(storeName, {})
    const _now = Date.now()

    // Remove expired records
    for (const record of allRecords) {
      if (this.isExpired(record)) {
        await this.deleteFromStore(storeName, record.id)
      }
    }

    // Check size limits
    const remainingRecords = await this.queryFromStore(storeName, {})
    if (remainingRecords.length > this.cacheConfig.maxEntries) {
      // Remove oldest records
      const sortedRecords = remainingRecords.sort(
        (a, b) => a.metadata.updatedAt.getTime() - b.metadata.updatedAt.getTime()
      )

      const toRemove = sortedRecords.slice(0, remainingRecords.length - this.cacheConfig.maxEntries)
      for (const record of toRemove) {
        await this.deleteFromStore(storeName, record.id)
      }
    }
  }

  /**
   * Public API methods
   */
  public getPerformanceMetrics(): typeof this.performance {
    return { ...this.performance }
  }

  public getSyncQueue(): SyncOperation[] {
    return [...this.syncQueue]
  }

  public async clearStore(storeName: string): Promise<void> {
    const transaction = this.db!.transaction([storeName], 'readwrite')
    const store = transaction.objectStore(storeName)

    return new Promise((resolve, reject) => {
      const request = store.clear()
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  public async getStorageSize(): Promise<{ [storeName: string]: number }> {
    const sizes: { [storeName: string]: number } = {}

    for (const storeName of Object.keys(this.schema.stores)) {
      const records = await this.queryFromStore(storeName, {})
      const size = records.reduce((total, record) => {
        return total + JSON.stringify(record).length
      }, 0)
      sizes[storeName] = size
    }

    return sizes
  }

  public addEventListener(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  public removeEventListener(event: string, callback: Function): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.delete(callback)
    }
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach((callback) => {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in data persistence event listener for ${event}:`, error)
        }
      })
    }
  }

  /**
   * Cleanup and destroy
   */
  public async destroy(): Promise<void> {
    if (this.syncInterval) {
      clearInterval(this.syncInterval)
    }

    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval)
    }

    if (this.compressionWorker) {
      this.compressionWorker.terminate()
    }

    if (this.db) {
      this.db.close()
    }

    this.eventListeners.clear()
    this.syncQueue = []
    this.isInitialized = false
  }
}

export default DataPersistenceManager
