/**
 * Virtual Scrolling Component for Large Lists
 * Renders only visible items to maintain performance with large datasets
 */
export class VirtualScroller<T> {
  private container: HTMLElement;
  private scrollContainer: HTMLElement;
  private items: T[] = [];
  private visibleItems: Map<number, HTMLElement> = new Map();
  private itemHeight: number;
  private containerHeight: number;
  private scrollTop = 0;
  private renderItem: (item: T, index: number) => HTMLElement;
  private cleanupCallbacks: (() => void)[] = [];
  
  // Performance optimization
  private animationFrame: number | null = null;
  private lastScrollTime = 0;
  private readonly SCROLL_THROTTLE = 16; // ~60fps
  
  // Memory optimization
  private readonly MAX_RENDERED_ITEMS = 100;
  private readonly BUFFER_SIZE = 10; // Extra items above/below viewport

  constructor(
    container: HTMLElement,
    itemHeight: number,
    renderItem: (item: T, index: number) => HTMLElement
  ) {
    this.container = container;
    this.itemHeight = itemHeight;
    this.renderItem = renderItem;
    this.containerHeight = container.clientHeight;
    
    this.setupScrollContainer();
    this.attachEventListeners();
  }

  private setupScrollContainer(): void {
    this.scrollContainer = document.createElement('div');
    this.scrollContainer.style.cssText = `
      height: 100%;
      overflow-y: auto;
      position: relative;
    `;
    
    this.container.appendChild(this.scrollContainer);
  }

  private attachEventListeners(): void {
    const handleScroll = this.throttledScroll.bind(this);
    this.scrollContainer.addEventListener('scroll', handleScroll, { passive: true });
    
    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
      this.containerHeight = this.container.clientHeight;
      this.updateVisibleItems();
    });
    
    resizeObserver.observe(this.container);
    
    this.cleanupCallbacks.push(() => {
      this.scrollContainer.removeEventListener('scroll', handleScroll);
      resizeObserver.disconnect();
    });
  }

  private throttledScroll(): void {
    const now = Date.now();
    if (now - this.lastScrollTime < this.SCROLL_THROTTLE) {
      if (this.animationFrame) return;
      
      this.animationFrame = requestAnimationFrame(() => {
        this.handleScroll();
        this.animationFrame = null;
      });
      return;
    }
    
    this.lastScrollTime = now;
    this.handleScroll();
  }

  private handleScroll(): void {
    this.scrollTop = this.scrollContainer.scrollTop;
    this.updateVisibleItems();
  }

  private updateVisibleItems(): void {
    if (this.items.length === 0) return;

    // Calculate visible range
    const startIndex = Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - this.BUFFER_SIZE);
    const endIndex = Math.min(
      this.items.length - 1,
      Math.ceil((this.scrollTop + this.containerHeight) / this.itemHeight) + this.BUFFER_SIZE
    );

    // Remove items outside visible range
    for (const [index, element] of this.visibleItems) {
      if (index < startIndex || index > endIndex) {
        element.remove();
        this.visibleItems.delete(index);
      }
    }

    // Add new visible items
    for (let i = startIndex; i <= endIndex; i++) {
      if (!this.visibleItems.has(i) && this.items[i]) {
        const element = this.createItemElement(this.items[i], i);
        this.visibleItems.set(i, element);
        this.scrollContainer.appendChild(element);
      }
    }

    // Limit total rendered items for memory efficiency
    if (this.visibleItems.size > this.MAX_RENDERED_ITEMS) {
      const entries = Array.from(this.visibleItems.entries());
      const toRemove = entries.slice(0, entries.length - this.MAX_RENDERED_ITEMS);
      
      toRemove.forEach(([index, element]) => {
        element.remove();
        this.visibleItems.delete(index);
      });
    }
  }

  private createItemElement(item: T, index: number): HTMLElement {
    const element = this.renderItem(item, index);
    
    // Position absolutely for virtual scrolling
    element.style.cssText = `
      position: absolute;
      top: ${index * this.itemHeight}px;
      left: 0;
      right: 0;
      height: ${this.itemHeight}px;
      box-sizing: border-box;
    `;
    
    return element;
  }

  public setItems(items: T[]): void {
    this.items = items;
    
    // Update scroll container height
    const totalHeight = items.length * this.itemHeight;
    this.scrollContainer.style.height = `${totalHeight}px`;
    
    // Clear existing items
    this.visibleItems.forEach(element => element.remove());
    this.visibleItems.clear();
    
    // Re-render visible items
    this.updateVisibleItems();
  }

  public addItem(item: T): void {
    this.items.push(item);
    this.setItems(this.items);
  }

  public removeItem(index: number): void {
    if (index >= 0 && index < this.items.length) {
      this.items.splice(index, 1);
      this.setItems(this.items);
    }
  }

  public scrollToIndex(index: number): void {
    if (index >= 0 && index < this.items.length) {
      const targetScrollTop = index * this.itemHeight;
      this.scrollContainer.scrollTop = targetScrollTop;
    }
  }

  public scrollToTop(): void {
    this.scrollContainer.scrollTop = 0;
  }

  public scrollToBottom(): void {
    this.scrollContainer.scrollTop = this.items.length * this.itemHeight;
  }

  public getVisibleRange(): { start: number; end: number } {
    const startIndex = Math.floor(this.scrollTop / this.itemHeight);
    const endIndex = Math.min(
      this.items.length - 1,
      Math.ceil((this.scrollTop + this.containerHeight) / this.itemHeight)
    );
    
    return { start: startIndex, end: endIndex };
  }

  public destroy(): void {
    // Cancel any pending animation frame
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    // Run cleanup callbacks
    this.cleanupCallbacks.forEach(cleanup => cleanup());
    this.cleanupCallbacks = [];

    // Clear DOM
    this.visibleItems.forEach(element => element.remove());
    this.visibleItems.clear();
    
    if (this.scrollContainer.parentNode) {
      this.scrollContainer.parentNode.removeChild(this.scrollContainer);
    }

    // Clear references
    this.items = [];
  }
}

/**
 * Virtual Log Viewer - Specialized virtual scroller for log entries
 */
export class VirtualLogViewer {
  private scroller: VirtualScroller<any>;
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    
    this.scroller = new VirtualScroller(
      container,
      32, // 32px per log entry
      this.renderLogItem.bind(this)
    );
  }

  private renderLogItem(entry: any, index: number): HTMLElement {
    const element = document.createElement('div');
    element.className = `log-item log-${entry.level}`;
    
    const time = new Date(entry.timestamp).toLocaleTimeString();
    element.innerHTML = `
      <span class="log-time">${time}</span>
      <span class="log-level">${entry.level.toUpperCase()}</span>
      ${entry.component ? `<span class="log-component">${entry.component}</span>` : ''}
      <span class="log-message">${entry.message}</span>
    `;
    
    return element;
  }

  public setLogs(logs: any[]): void {
    this.scroller.setItems(logs);
  }

  public addLog(log: any): void {
    this.scroller.addItem(log);
    // Auto-scroll to bottom for new logs
    this.scroller.scrollToBottom();
  }

  public destroy(): void {
    this.scroller.destroy();
  }
}