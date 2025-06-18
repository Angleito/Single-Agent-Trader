/**
 * Mobile Optimization and Responsive Design Manager
 *
 * Provides comprehensive mobile optimization features:
 * - Responsive layout management and adaptation
 * - Touch gesture recognition and handling
 * - Mobile-specific UI optimizations
 * - Performance optimization for mobile devices
 * - Offline-first mobile experience
 * - Native app-like behavior and interactions
 * - Accessibility improvements for mobile
 * - Mobile-specific error handling
 */

export interface MobileConfig {
  breakpoints: {
    mobile: number
    tablet: number
    desktop: number
    largeDesktop: number
  }
  touchEnabled: boolean
  gesturesEnabled: boolean
  vibrationEnabled: boolean
  orientationLockEnabled: boolean
  fullscreenEnabled: boolean
  statusBarStyle: 'default' | 'light' | 'dark'
  optimizations: {
    enableLazyLoading: boolean
    enableImageOptimization: boolean
    enableTextCompression: boolean
    enableAnimationReduction: boolean
    enableBatteryOptimization: boolean
  }
}

export interface DeviceInfo {
  type: 'mobile' | 'tablet' | 'desktop'
  os: 'ios' | 'android' | 'windows' | 'macos' | 'linux' | 'unknown'
  browser: 'chrome' | 'safari' | 'firefox' | 'edge' | 'unknown'
  isTouch: boolean
  screenSize: 'small' | 'medium' | 'large' | 'xlarge'
  orientation: 'portrait' | 'landscape'
  pixelRatio: number
  battery?: {
    level: number
    charging: boolean
  }
  connection?: {
    effectiveType: string
    downlink: number
    rtt: number
  }
}

export interface GestureEvent {
  type: 'tap' | 'double-tap' | 'long-press' | 'swipe' | 'pinch' | 'rotate'
  target: HTMLElement
  startPoint: { x: number; y: number }
  endPoint: { x: number; y: number }
  deltaX: number
  deltaY: number
  distance: number
  duration: number
  direction?: 'up' | 'down' | 'left' | 'right'
  scale?: number
  rotation?: number
}

export interface LayoutState {
  currentBreakpoint: string
  screenSize: string
  orientation: string
  availableHeight: number
  availableWidth: number
  safeAreaInsets: {
    top: number
    right: number
    bottom: number
    left: number
  }
  keyboardHeight: number
  isFullscreen: boolean
}

export class MobileOptimizationManager {
  private config: MobileConfig
  private deviceInfo: DeviceInfo
  private layoutState: LayoutState
  private eventListeners = new Map<string, Set<(...args: any[]) => void>>()
  private gestureHandlers = new Map<string, (...args: any[]) => void>()
  private resizeObserver: ResizeObserver | null = null
  private intersectionObserver: IntersectionObserver | null = null
  private performanceMonitor: PerformanceObserver | null = null
  private wakeLock: any = null
  private isInitialized = false

  // Touch and gesture tracking
  private touchStartTime = 0
  private touchStartPoint = { x: 0, y: 0 }
  private lastTouchEnd = 0
  private touchCount = 0
  private gestureStartDistance = 0
  private gestureStartAngle = 0

  // Performance optimization
  private animationFrameId: number | null = null
  private throttledUpdates = new Map<string, number>()
  private lazyElements = new Set<HTMLElement>()

  constructor(config: Partial<MobileConfig>) {
    this.config = {
      breakpoints: {
        mobile: 768,
        tablet: 1024,
        desktop: 1440,
        largeDesktop: 1920,
      },
      touchEnabled: true,
      gesturesEnabled: true,
      vibrationEnabled: true,
      orientationLockEnabled: false,
      fullscreenEnabled: false,
      statusBarStyle: 'dark',
      optimizations: {
        enableLazyLoading: true,
        enableImageOptimization: true,
        enableTextCompression: true,
        enableAnimationReduction: false,
        enableBatteryOptimization: true,
      },
      ...config,
    }

    this.deviceInfo = this.detectDevice()
    this.layoutState = this.initializeLayoutState()

    void this.init()
  }

  /**
   * Initialize the mobile optimization manager
   */
  private init(): void {
    if (this.isInitialized) return

    try {
      // Setup device detection
      this.setupDeviceListeners()

      // Setup responsive layout
      this.setupResponsiveLayout()

      // Setup touch and gesture handling
      if (this.config.touchEnabled) {
        this.setupTouchHandling()
      }

      // Setup performance optimizations
      this.setupPerformanceOptimizations()

      // Setup PWA features
      this.setupPWAFeatures()

      // Setup accessibility
      this.setupAccessibility()

      // Apply initial optimizations
      this.applyMobileOptimizations()

      this.isInitialized = true
      this.emit('initialized', { deviceInfo: this.deviceInfo, layoutState: this.layoutState })
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('[MobileOptimization] Initialization failed:', error)
      throw error
    }
  }

  /**
   * Detect device information
   */
  private detectDevice(): DeviceInfo {
    const userAgent = navigator.userAgent
    const screenWidth = window.screen.width
    const screenHeight = window.screen.height
    const pixelRatio = window.devicePixelRatio || 1

    // Detect device type
    let type: 'mobile' | 'tablet' | 'desktop' = 'desktop'
    if (screenWidth <= this.config.breakpoints.mobile) {
      type = 'mobile'
    } else if (screenWidth <= this.config.breakpoints.tablet) {
      type = 'tablet'
    }

    // Detect OS
    let os: DeviceInfo['os'] = 'unknown'
    if (/iPad|iPhone|iPod/.test(userAgent)) {
      os = 'ios'
    } else if (/Android/.test(userAgent)) {
      os = 'android'
    } else if (/Windows/.test(userAgent)) {
      os = 'windows'
    } else if (/Mac/.test(userAgent)) {
      os = 'macos'
    } else if (/Linux/.test(userAgent)) {
      os = 'linux'
    }

    // Detect browser
    let browser: DeviceInfo['browser'] = 'unknown'
    if (/Chrome/.test(userAgent)) {
      browser = 'chrome'
    } else if (/Safari/.test(userAgent)) {
      browser = 'safari'
    } else if (/Firefox/.test(userAgent)) {
      browser = 'firefox'
    } else if (/Edge/.test(userAgent)) {
      browser = 'edge'
    }

    // Detect screen size
    let screenSize: DeviceInfo['screenSize'] = 'large'
    if (screenWidth <= 480) {
      screenSize = 'small'
    } else if (screenWidth <= 768) {
      screenSize = 'medium'
    } else if (screenWidth <= 1440) {
      screenSize = 'large'
    } else {
      screenSize = 'xlarge'
    }

    // Detect orientation
    const orientation: DeviceInfo['orientation'] =
      screenWidth > screenHeight ? 'landscape' : 'portrait'

    const deviceInfo: DeviceInfo = {
      type,
      os,
      browser,
      isTouch: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
      screenSize,
      orientation,
      pixelRatio,
    }

    // Get battery info if available
    if ('getBattery' in navigator) {
      void (navigator as any).getBattery().then((battery: any) => {
        deviceInfo.battery = {
          level: battery.level,
          charging: battery.charging,
        }
      })
    }

    // Get connection info if available
    if ('connection' in navigator) {
      const connection = (navigator as any).connection
      deviceInfo.connection = {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
      }
    }

    return deviceInfo
  }

  /**
   * Initialize layout state
   */
  private initializeLayoutState(): LayoutState {
    const viewport = this.getViewportDimensions()

    return {
      currentBreakpoint: this.getCurrentBreakpoint(),
      screenSize: this.deviceInfo.screenSize,
      orientation: this.deviceInfo.orientation,
      availableHeight: viewport.height,
      availableWidth: viewport.width,
      safeAreaInsets: this.getSafeAreaInsets(),
      keyboardHeight: 0,
      isFullscreen: this.isFullscreen(),
    }
  }

  /**
   * Setup device event listeners
   */
  private setupDeviceListeners(): void {
    // Orientation change
    window.addEventListener('orientationchange', () => {
      setTimeout(() => {
        this.handleOrientationChange()
      }, 100) // Small delay to allow for orientation change completion
    })

    // Resize
    window.addEventListener(
      'resize',
      this.throttle(() => {
        this.handleResize()
      }, 250) as EventListener
    )

    // Battery status change
    if ('getBattery' in navigator) {
      void (navigator as any).getBattery().then((battery: any) => {
        battery.addEventListener('levelchange', () => {
          this.updateBatteryInfo()
        })
        battery.addEventListener('chargingchange', () => {
          this.updateBatteryInfo()
        })
      })
    }

    // Network change
    if ('connection' in navigator) {
      const connection = (navigator as any).connection
      connection.addEventListener('change', () => {
        this.updateConnectionInfo()
      })
    }

    // Visibility change
    document.addEventListener('visibilitychange', () => {
      this.handleVisibilityChange()
    })
  }

  /**
   * Setup responsive layout management
   */
  private setupResponsiveLayout(): void {
    // Setup ResizeObserver for component-level responsiveness
    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          this.handleElementResize(entry)
        }
      })
    }

    // Apply initial responsive classes
    this.updateResponsiveClasses()

    // Setup CSS custom properties for dynamic sizing
    this.updateCSSProperties()
  }

  /**
   * Setup touch and gesture handling
   */
  private setupTouchHandling(): void {
    if (!this.deviceInfo.isTouch) return

    // Disable default behaviors that interfere with gestures
    document.addEventListener(
      'touchmove',
      (e) => {
        // Prevent pull-to-refresh and overscroll
        if (this.shouldPreventDefault(e)) {
          e.preventDefault()
        }
      },
      { passive: false }
    )

    // Touch events
    document.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: true })
    document.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: true })
    document.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: true })
    document.addEventListener('touchcancel', (e) => this.handleTouchCancel(e), { passive: true })

    // Pointer events for more advanced gesture recognition
    if ('PointerEvent' in window) {
      document.addEventListener('pointerdown', (e) => this.handlePointerDown(e))
      document.addEventListener('pointermove', (e) => this.handlePointerMove(e))
      document.addEventListener('pointerup', (e) => this.handlePointerUp(e))
    }

    // Prevent context menu on long press for better UX
    document.addEventListener('contextmenu', (e) => {
      if (this.deviceInfo.isTouch) {
        e.preventDefault()
      }
    })
  }

  /**
   * Setup performance optimizations
   */
  private setupPerformanceOptimizations(): void {
    // Lazy loading
    if (this.config.optimizations.enableLazyLoading) {
      this.setupLazyLoading()
    }

    // Performance monitoring
    if ('PerformanceObserver' in window) {
      this.performanceMonitor = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        this.handlePerformanceEntries(entries)
      })

      this.performanceMonitor.observe({
        entryTypes: ['measure', 'navigation', 'largest-contentful-paint'],
      })
    }

    // Memory management
    this.setupMemoryManagement()

    // Animation optimization
    if (this.config.optimizations.enableAnimationReduction) {
      this.setupAnimationOptimization()
    }
  }

  /**
   * Setup PWA features
   */
  private setupPWAFeatures(): void {
    // Install prompt handling
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault()
      this.emit('installPrompt', e)
    })

    // App installed
    window.addEventListener('appinstalled', () => {
      this.emit('appInstalled')
    })

    // Standalone mode detection
    if (window.matchMedia('(display-mode: standalone)').matches) {
      document.body.classList.add('standalone-mode')
      this.emit('standaloneMode', true)
    }

    // Status bar styling for iOS
    if (this.deviceInfo.os === 'ios') {
      this.setupiOSStatusBar()
    }
  }

  /**
   * Setup accessibility features
   */
  private setupAccessibility(): void {
    // Reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.body.classList.add('reduced-motion')
      this.config.optimizations.enableAnimationReduction = true
    }

    // High contrast preference
    if (window.matchMedia('(prefers-contrast: high)').matches) {
      document.body.classList.add('high-contrast')
    }

    // Focus management for mobile
    this.setupFocusManagement()

    // Screen reader optimizations
    this.setupScreenReaderOptimizations()
  }

  /**
   * Apply mobile-specific optimizations
   */
  private applyMobileOptimizations(): void {
    const body = document.body

    // Add device-specific classes
    body.classList.add(`device-${this.deviceInfo.type}`)
    body.classList.add(`os-${this.deviceInfo.os}`)
    body.classList.add(`browser-${this.deviceInfo.browser}`)

    if (this.deviceInfo.isTouch) {
      body.classList.add('touch-device')
    }

    // Disable zoom on mobile if needed
    if (this.deviceInfo.type === 'mobile') {
      this.disableZoom()
    }

    // Set viewport height for mobile browsers
    this.setViewportHeight()

    // Optimize scrolling
    this.optimizeScrolling()

    // Setup safe area handling
    this.setupSafeAreaHandling()
  }

  /**
   * Handle touch events
   */
  private handleTouchStart(event: TouchEvent): void {
    this.touchStartTime = Date.now()
    this.touchCount = event.touches.length

    if (event.touches.length === 1) {
      const touch = event.touches[0]
      this.touchStartPoint = { x: touch.clientX, y: touch.clientY }
    } else if (event.touches.length === 2) {
      // Multi-touch gesture start
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]

      this.gestureStartDistance = this.calculateDistance(
        touch1.clientX,
        touch1.clientY,
        touch2.clientX,
        touch2.clientY
      )

      this.gestureStartAngle = this.calculateAngle(
        touch1.clientX,
        touch1.clientY,
        touch2.clientX,
        touch2.clientY
      )
    }
  }

  private handleTouchMove(event: TouchEvent): void {
    if (!this.config.gesturesEnabled) return

    if (event.touches.length === 2) {
      // Handle pinch/zoom gesture
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]

      const currentDistance = this.calculateDistance(
        touch1.clientX,
        touch1.clientY,
        touch2.clientX,
        touch2.clientY
      )

      const currentAngle = this.calculateAngle(
        touch1.clientX,
        touch1.clientY,
        touch2.clientX,
        touch2.clientY
      )

      const scale = currentDistance / this.gestureStartDistance
      const rotation = currentAngle - this.gestureStartAngle

      this.emit('gesture', {
        type: 'pinch',
        scale,
        rotation,
        target: event.target as HTMLElement,
      })
    }
  }

  private handleTouchEnd(event: TouchEvent): void {
    const touchEndTime = Date.now()
    const touchDuration = touchEndTime - this.touchStartTime

    if (event.changedTouches.length === 1 && this.touchCount === 1) {
      const touch = event.changedTouches[0]
      const endPoint = { x: touch.clientX, y: touch.clientY }

      const deltaX = endPoint.x - this.touchStartPoint.x
      const deltaY = endPoint.y - this.touchStartPoint.y
      const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)

      // Determine gesture type
      if (distance < 10 && touchDuration < 300) {
        // Tap
        const timeSinceLastTap = touchEndTime - this.lastTouchEnd
        if (timeSinceLastTap < 300) {
          // Double tap
          this.emitGesture('double-tap', event.target as HTMLElement, {
            startPoint: this.touchStartPoint,
            endPoint,
            deltaX,
            deltaY,
            distance,
            duration: touchDuration,
          })
        } else {
          // Single tap
          this.emitGesture('tap', event.target as HTMLElement, {
            startPoint: this.touchStartPoint,
            endPoint,
            deltaX,
            deltaY,
            distance,
            duration: touchDuration,
          })
        }

        this.lastTouchEnd = touchEndTime
      } else if (distance < 10 && touchDuration >= 500) {
        // Long press
        this.emitGesture('long-press', event.target as HTMLElement, {
          startPoint: this.touchStartPoint,
          endPoint,
          deltaX,
          deltaY,
          distance,
          duration: touchDuration,
        })

        // Haptic feedback
        this.vibrate(50)
      } else if (distance >= 30) {
        // Swipe
        const direction = this.getSwipeDirection(deltaX, deltaY)
        this.emitGesture('swipe', event.target as HTMLElement, {
          startPoint: this.touchStartPoint,
          endPoint,
          deltaX,
          deltaY,
          distance,
          duration: touchDuration,
          direction,
        })
      }
    }
  }

  private handleTouchCancel(_event: TouchEvent): void {
    // Reset touch state
    this.touchStartTime = 0
    this.touchCount = 0
  }

  /**
   * Handle layout changes
   */
  private handleOrientationChange(): void {
    this.deviceInfo.orientation = window.innerWidth > window.innerHeight ? 'landscape' : 'portrait'
    this.layoutState.orientation = this.deviceInfo.orientation

    // Update layout
    this.updateLayout()

    // Emit orientation change event
    this.emit('orientationChange', this.deviceInfo.orientation)
  }

  private handleResize(): void {
    this.updateLayout()
    this.updateCSSProperties()
    this.emit('resize', this.layoutState)
  }

  private updateLayout(): void {
    const viewport = this.getViewportDimensions()
    const newBreakpoint = this.getCurrentBreakpoint()

    // Update layout state
    this.layoutState.availableWidth = viewport.width
    this.layoutState.availableHeight = viewport.height
    this.layoutState.safeAreaInsets = this.getSafeAreaInsets()

    if (newBreakpoint !== this.layoutState.currentBreakpoint) {
      this.layoutState.currentBreakpoint = newBreakpoint
      this.updateResponsiveClasses()
      this.emit('breakpointChange', newBreakpoint)
    }

    // Update viewport height
    this.setViewportHeight()
  }

  /**
   * Utility methods
   */
  private getCurrentBreakpoint(): string {
    const width = window.innerWidth

    if (width <= this.config.breakpoints.mobile) {
      return 'mobile'
    } else if (width <= this.config.breakpoints.tablet) {
      return 'tablet'
    } else if (width <= this.config.breakpoints.desktop) {
      return 'desktop'
    } else {
      return 'large-desktop'
    }
  }

  private getViewportDimensions() {
    return {
      width: window.innerWidth,
      height: window.innerHeight,
    }
  }

  private getSafeAreaInsets() {
    const style = getComputedStyle(document.documentElement)

    return {
      top: parseInt(style.getPropertyValue('--sat') || '0'),
      right: parseInt(style.getPropertyValue('--sar') || '0'),
      bottom: parseInt(style.getPropertyValue('--sab') || '0'),
      left: parseInt(style.getPropertyValue('--sal') || '0'),
    }
  }

  private isFullscreen(): boolean {
    return !!(
      document.fullscreenElement ??
      (document as any).webkitFullscreenElement ??
      (document as any).mozFullScreenElement
    )
  }

  private updateResponsiveClasses(): void {
    const body = document.body

    // Remove old breakpoint classes
    body.className = body.className.replace(/\bbreakpoint-\w+/g, '')

    // Add current breakpoint class
    body.classList.add(`breakpoint-${this.layoutState.currentBreakpoint}`)
    body.classList.add(`orientation-${this.layoutState.orientation}`)
  }

  private updateCSSProperties(): void {
    const root = document.documentElement

    root.style.setProperty('--viewport-width', `${this.layoutState.availableWidth}px`)
    root.style.setProperty('--viewport-height', `${this.layoutState.availableHeight}px`)
    root.style.setProperty('--safe-area-inset-top', `${this.layoutState.safeAreaInsets.top}px`)
    root.style.setProperty('--safe-area-inset-right', `${this.layoutState.safeAreaInsets.right}px`)
    root.style.setProperty(
      '--safe-area-inset-bottom',
      `${this.layoutState.safeAreaInsets.bottom}px`
    )
    root.style.setProperty('--safe-area-inset-left', `${this.layoutState.safeAreaInsets.left}px`)
  }

  private setViewportHeight(): void {
    // Fix for mobile browser viewport height issues
    const vh = window.innerHeight * 0.01
    document.documentElement.style.setProperty('--vh', `${vh}px`)
  }

  private disableZoom(): void {
    const viewport = document.querySelector('meta[name="viewport"]')
    if (viewport) {
      viewport.setAttribute(
        'content',
        'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
      )
    }
  }

  private optimizeScrolling(): void {
    // Enable momentum scrolling on iOS
    ;(document.body.style as any).webkitOverflowScrolling = 'touch'

    // Optimize scroll performance
    document.addEventListener(
      'touchstart',
      (e) => {
        // Enable hardware acceleration for scrolling elements
        const target = e.target as HTMLElement
        if (target && target.scrollHeight > target.clientHeight) {
          target.style.transform = 'translateZ(0)'
        }
      },
      { passive: true }
    )
  }

  private setupSafeAreaHandling(): void {
    // iOS safe area support
    if (this.deviceInfo.os === 'ios') {
      const root = document.documentElement
      root.style.setProperty('--sat', 'env(safe-area-inset-top)')
      root.style.setProperty('--sar', 'env(safe-area-inset-right)')
      root.style.setProperty('--sab', 'env(safe-area-inset-bottom)')
      root.style.setProperty('--sal', 'env(safe-area-inset-left)')
    }
  }

  private setupiOSStatusBar(): void {
    const metaTag = document.createElement('meta')
    metaTag.name = 'apple-mobile-web-app-status-bar-style'
    metaTag.content = this.config.statusBarStyle
    document.head.appendChild(metaTag)
  }

  private setupLazyLoading(): void {
    if ('IntersectionObserver' in window) {
      this.intersectionObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              this.loadLazyElement(entry.target as HTMLElement)
            }
          })
        },
        {
          rootMargin: '50px',
        }
      )
    }
  }

  private loadLazyElement(element: HTMLElement): void {
    // Load lazy images
    if (element.tagName === 'IMG') {
      const img = element as HTMLImageElement
      const dataSrc = img.dataset.src
      if (dataSrc) {
        img.src = dataSrc
        img.removeAttribute('data-src')
      }
    }

    // Load lazy content
    const lazyContent = element.dataset.lazyContent
    if (lazyContent) {
      element.innerHTML = lazyContent
      element.removeAttribute('data-lazy-content')
    }

    this.intersectionObserver?.unobserve(element)
    this.lazyElements.delete(element)
  }

  private setupMemoryManagement(): void {
    // Clean up unused resources periodically
    setInterval(() => {
      this.cleanupResources()
    }, 60000) // Every minute
  }

  private cleanupResources(): void {
    // Remove throttled updates that are old
    const now = Date.now()
    for (const [key, timestamp] of this.throttledUpdates.entries()) {
      if (now - timestamp > 30000) {
        // 30 seconds
        this.throttledUpdates.delete(key)
      }
    }
  }

  private setupAnimationOptimization(): void {
    const style = document.createElement('style')
    style.textContent = `
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    `
    document.head.appendChild(style)
  }

  private setupFocusManagement(): void {
    // Improve focus visibility for keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        document.body.classList.add('keyboard-navigation')
      }
    })

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('keyboard-navigation')
    })
  }

  private setupScreenReaderOptimizations(): void {
    // Add screen reader only content for important information
    const style = document.createElement('style')
    style.textContent = `
      .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }
    `
    document.head.appendChild(style)
  }

  /**
   * Gesture utilities
   */
  private calculateDistance(x1: number, y1: number, x2: number, y2: number): number {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
  }

  private calculateAngle(x1: number, y1: number, x2: number, y2: number): number {
    return (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI
  }

  private getSwipeDirection(deltaX: number, deltaY: number): 'up' | 'down' | 'left' | 'right' {
    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      return deltaX > 0 ? 'right' : 'left'
    } else {
      return deltaY > 0 ? 'down' : 'up'
    }
  }

  private emitGesture(type: string, target: HTMLElement, data: any): void {
    const gestureEvent: GestureEvent = {
      type: type as any,
      target,
      ...data,
    }

    this.emit('gesture', gestureEvent)

    // Call specific gesture handler if registered
    const handler = this.gestureHandlers.get(type)
    if (handler) {
      handler(gestureEvent)
    }
  }

  private shouldPreventDefault(event: TouchEvent): boolean {
    const target = event.target as HTMLElement

    // Don't prevent default for form inputs
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return false
    }

    // Prevent default for scrollable elements at boundaries
    if (target.scrollHeight > target.clientHeight) {
      const isAtTop = target.scrollTop === 0
      const isAtBottom = target.scrollTop >= target.scrollHeight - target.clientHeight

      if (
        (isAtTop && event.touches[0].clientY > 0) ||
        (isAtBottom && event.touches[0].clientY < 0)
      ) {
        return true
      }
    }

    return false
  }

  private throttle(func: (...args: any[]) => void, delay: number): (...args: any[]) => void {
    let timeoutId: number
    let lastExecTime = 0

    return function (this: any, ...args: any[]) {
      const currentTime = Date.now()

      if (currentTime - lastExecTime > delay) {
        func.apply(this, args)
        lastExecTime = currentTime
      } else {
        clearTimeout(timeoutId)
        timeoutId = window.setTimeout(
          () => {
            func.apply(this, args)
            lastExecTime = Date.now()
          },
          delay - (currentTime - lastExecTime)
        )
      }
    }
  }

  private vibrate(duration: number | number[]): void {
    if (this.config.vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate(duration)
    }
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach((callback) => {
        try {
          callback(data)
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error(`[MobileOptimization] Error in event listener for ${event}:`, error)
        }
      })
    }
  }

  private handleElementResize(entry: ResizeObserverEntry): void {
    // Handle component-specific resize logic
    const element = entry.target as HTMLElement
    const componentId = element.dataset.componentId

    if (componentId) {
      this.emit('componentResize', {
        componentId,
        element,
        contentRect: entry.contentRect,
      })
    }
  }

  private handlePerformanceEntries(entries: PerformanceEntry[]): void {
    // Monitor performance and adapt accordingly
    entries.forEach((entry) => {
      if (entry.entryType === 'largest-contentful-paint') {
        const lcp = entry as PerformanceNavigationTiming
        if (lcp.duration > 2500) {
          // Poor LCP, enable more aggressive optimizations
          this.enableAggressiveOptimizations()
        }
      }
    })
  }

  private enableAggressiveOptimizations(): void {
    // Reduce image quality
    const images = document.querySelectorAll('img')
    images.forEach((img) => {
      if (!img.dataset.optimized) {
        img.loading = 'lazy'
        img.dataset.optimized = 'true'
      }
    })

    // Reduce animation complexity
    document.body.classList.add('performance-mode')
  }

  private handleVisibilityChange(): void {
    if (document.hidden) {
      // App is in background, reduce activity
      this.pauseNonEssentialActivity()
    } else {
      // App is in foreground, resume activity
      this.resumeActivity()
    }
  }

  private pauseNonEssentialActivity(): void {
    // Cancel animation frames
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId)
    }

    this.emit('backgrounded')
  }

  private resumeActivity(): void {
    this.emit('foregrounded')
  }

  private updateBatteryInfo(): void {
    if ('getBattery' in navigator) {
      void (navigator as any).getBattery().then((battery: any) => {
        this.deviceInfo.battery = {
          level: battery.level,
          charging: battery.charging,
        }

        // Enable battery optimizations if low
        if (battery.level < 0.2 && !battery.charging) {
          this.enableBatteryOptimizations()
        }

        this.emit('batteryChange', this.deviceInfo.battery)
      })
    }
  }

  private updateConnectionInfo(): void {
    if ('connection' in navigator) {
      const connection = (navigator as any).connection
      this.deviceInfo.connection = {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
      }

      // Adapt to slow connections
      if (connection.effectiveType === 'slow-2g' || connection.effectiveType === '2g') {
        this.enableSlowConnectionOptimizations()
      }

      this.emit('connectionChange', this.deviceInfo.connection)
    }
  }

  private enableBatteryOptimizations(): void {
    document.body.classList.add('battery-saver')
    this.config.optimizations.enableAnimationReduction = true
    this.emit('batteryOptimizationsEnabled')
  }

  private enableSlowConnectionOptimizations(): void {
    document.body.classList.add('slow-connection')
    this.config.optimizations.enableImageOptimization = true
    this.emit('slowConnectionOptimizationsEnabled')
  }

  /**
   * Pointer event handlers for advanced gesture recognition
   */
  private handlePointerDown(_event: PointerEvent): void {
    // Enhanced gesture recognition with pointer events
  }

  private handlePointerMove(_event: PointerEvent): void {
    // Track pointer movements for gestures
  }

  private handlePointerUp(_event: PointerEvent): void {
    // Complete gesture recognition
  }

  /**
   * Public API methods
   */
  public addEventListener(event: string, callback: (...args: any[]) => void): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  public removeEventListener(event: string, callback: (...args: any[]) => void): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventListeners.delete(event)
      }
    }
  }

  public registerGestureHandler(gestureType: string, handler: (...args: any[]) => void): void {
    this.gestureHandlers.set(gestureType, handler)
  }

  public unregisterGestureHandler(gestureType: string): void {
    this.gestureHandlers.delete(gestureType)
  }

  public observeElement(element: HTMLElement): void {
    if (this.resizeObserver) {
      this.resizeObserver.observe(element)
    }
  }

  public unobserveElement(element: HTMLElement): void {
    if (this.resizeObserver) {
      this.resizeObserver.unobserve(element)
    }
  }

  public addLazyElement(element: HTMLElement): void {
    if (this.intersectionObserver) {
      this.intersectionObserver.observe(element)
      this.lazyElements.add(element)
    }
  }

  public removeLazyElement(element: HTMLElement): void {
    if (this.intersectionObserver) {
      this.intersectionObserver.unobserve(element)
      this.lazyElements.delete(element)
    }
  }

  public getDeviceInfo(): DeviceInfo {
    return { ...this.deviceInfo }
  }

  public getLayoutState(): LayoutState {
    return { ...this.layoutState }
  }

  public updateConfig(newConfig: Partial<MobileConfig>): void {
    this.config = { ...this.config, ...newConfig }
    this.emit('configUpdated', this.config)
  }

  public requestFullscreen(): Promise<void> {
    if (document.documentElement.requestFullscreen) {
      return document.documentElement.requestFullscreen()
    }
    return Promise.reject(new Error('Fullscreen not supported'))
  }

  public exitFullscreen(): Promise<void> {
    if (document.exitFullscreen) {
      return document.exitFullscreen()
    }
    return Promise.reject(new Error('Exit fullscreen not supported'))
  }

  public requestWakeLock(): Promise<void> {
    if ('wakeLock' in navigator) {
      return (navigator as any).wakeLock.request('screen').then((wakeLock: any) => {
        this.wakeLock = wakeLock
        this.emit('wakeLockAcquired')
      })
    }
    return Promise.reject(new Error('Wake lock not supported'))
  }

  public releaseWakeLock(): void {
    if (this.wakeLock) {
      this.wakeLock.release()
      this.wakeLock = null
      this.emit('wakeLockReleased')
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    // Clean up observers
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
    }

    if (this.intersectionObserver) {
      this.intersectionObserver.disconnect()
    }

    if (this.performanceMonitor) {
      this.performanceMonitor.disconnect()
    }

    // Release wake lock
    this.releaseWakeLock()

    // Cancel animation frames
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId)
    }

    // Clear event listeners
    this.eventListeners.clear()
    this.gestureHandlers.clear()

    // Clear collections
    this.lazyElements.clear()
    this.throttledUpdates.clear()
  }
}
