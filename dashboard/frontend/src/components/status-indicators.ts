// Simple status indicators component without Lit dependencies

// Types for status indicators
export interface SystemHealth {
  cpu?: number;
  memory?: number;
  uptime?: number;
}

export interface AlertMessage {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  sound?: boolean;
}

export interface ConnectionStatus {
  websocket: 'connected' | 'disconnected' | 'reconnecting';
  latency?: number;
  lastHeartbeat?: Date;
}

export interface BotStatus {
  state: 'active' | 'paused' | 'error' | 'initializing';
  message?: string;
}

export interface MarketStatus {
  state: 'open' | 'closed' | 'pre-market' | 'after-hours';
  nextChange?: Date;
}

export interface PositionStatus {
  state: 'in-position' | 'flat' | 'pending-entry' | 'pending-exit';
  count?: number;
}

export class StatusIndicators {
  public connectionStatus: ConnectionStatus = {
    websocket: 'disconnected'
  };
  
  public botStatus: BotStatus = {
    state: 'initializing'
  };
  
  public marketStatus: MarketStatus = {
    state: 'closed'
  };
  
  public positionStatus: PositionStatus = {
    state: 'flat'
  };
  
  public systemHealth: SystemHealth = {};
  
  public alerts: AlertMessage[] = [];
  public soundEnabled = true;
  public showAlertHistory = false;

  private alertContainer: HTMLElement | null = null;
  private audioContext?: AudioContext;
  private audioInitialized = false;

  constructor() {
    this.createAlertContainer();
    this.setupAudioInitialization();
  }

  private setupAudioInitialization() {
    // Set up one-time user interaction listener for audio context
    const initAudio = () => {
      if (!this.audioInitialized && typeof window !== 'undefined' && 'AudioContext' in window) {
        try {
          this.audioContext = new AudioContext();
          this.audioInitialized = true;
          // Remove listeners once initialized
          document.removeEventListener('click', initAudio);
          document.removeEventListener('keydown', initAudio);
          document.removeEventListener('touchstart', initAudio);
        } catch (error) {
          console.warn('Failed to initialize AudioContext:', error);
        }
      }
    };

    // Add listeners for user interaction
    document.addEventListener('click', initAudio, { once: true, passive: true });
    document.addEventListener('keydown', initAudio, { once: true, passive: true });
    document.addEventListener('touchstart', initAudio, { once: true, passive: true });
  }

  private createAlertContainer() {
    // Create alert container for toasts
    const container = document.createElement('div');
    container.className = 'alert-container';
    container.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-width: 400px;
      pointer-events: none;
    `;
    document.body.appendChild(container);
    this.alertContainer = container;
  }

  private async playAlertSound(type: 'success' | 'warning' | 'error') {
    if (!this.soundEnabled || !this.audioContext) return;

    // Resume audio context if suspended (required by some browsers)
    if (this.audioContext.state === 'suspended') {
      try {
        await this.audioContext.resume();
      } catch (error) {
        console.warn('Failed to resume AudioContext:', error);
        return;
      }
    }

    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(this.audioContext.destination);

    // Different frequencies for different alert types
    const frequencies = {
      success: [523.25, 659.25], // C5, E5
      warning: [440, 554.37], // A4, C#5
      error: [329.63, 293.66] // E4, D4
    };

    const freq = frequencies[type] || frequencies.warning;
    
    oscillator.frequency.setValueAtTime(freq[0], this.audioContext.currentTime);
    oscillator.frequency.setValueAtTime(freq[1], this.audioContext.currentTime + 0.1);
    
    gainNode.gain.setValueAtTime(0.3, this.audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.2);

    oscillator.start(this.audioContext.currentTime);
    oscillator.stop(this.audioContext.currentTime + 0.2);
  }

  addAlert(alert: Omit<AlertMessage, 'id' | 'timestamp'>) {
    const newAlert: AlertMessage = {
      ...alert,
      id: Date.now().toString(),
      timestamp: new Date()
    };

    this.alerts = [newAlert, ...this.alerts].slice(0, 50); // Keep last 50 alerts

    if (alert.sound && (alert.type === 'success' || alert.type === 'warning' || alert.type === 'error')) {
      this.playAlertSound(alert.type);
    }

    // Show the alert toast
    this.showAlertToast(newAlert);

    // Auto-remove toast after 5 seconds
    setTimeout(() => {
      this.removeAlert(newAlert.id);
    }, 5000);
  }

  private showAlertToast(alert: AlertMessage) {
    if (!this.alertContainer) return;

    const toastElement = document.createElement('div');
    toastElement.className = `alert-toast ${alert.type}`;
    toastElement.style.cssText = `
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 16px;
      background: #1f2937;
      border-radius: 8px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.1);
      animation: slide-in 0.3s ease-out;
      pointer-events: auto;
      cursor: pointer;
    `;
    toastElement.setAttribute('data-alert-id', alert.id);

    const iconMap = {
      info: 'ℹ️',
      success: '✅',
      warning: '⚠️',
      error: '❌'
    };

    toastElement.innerHTML = `
      <span style="font-size: 20px; flex-shrink: 0;">${iconMap[alert.type]}</span>
      <div style="flex: 1;">
        <div style="font-weight: 600; margin-bottom: 4px;">${alert.title}</div>
        <div style="font-size: 14px; color: #9ca3af;">${alert.message}</div>
      </div>
      <button style="background: none; border: none; color: #6b7280; cursor: pointer; padding: 4px; font-size: 16px;">✕</button>
    `;

    // Close button functionality
    const closeBtn = toastElement.querySelector('button');
    closeBtn?.addEventListener('click', () => this.removeAlert(alert.id));

    // Click to dismiss
    toastElement.addEventListener('click', () => this.removeAlert(alert.id));

    this.alertContainer.appendChild(toastElement);
  }

  private removeAlert(id: string) {
    const alertElement = this.alertContainer?.querySelector(`[data-alert-id="${id}"]`) as HTMLElement;
    if (alertElement) {
      alertElement.style.animation = 'slide-out 0.3s ease-in forwards';
      setTimeout(() => {
        alertElement.remove();
        this.alerts = this.alerts.filter(a => a.id !== id);
      }, 300);
    }
  }

  public formatLatency(latency?: number): string {
    if (!latency) return '';
    if (latency < 100) return `${latency}ms`;
    if (latency < 1000) return `${latency}ms`;
    return `${(latency / 1000).toFixed(1)}s`;
  }

  public getLatencyClass(latency?: number): string {
    if (!latency) return '';
    if (latency < 100) return 'latency-good';
    if (latency < 500) return 'latency-medium';
    return 'latency-poor';
  }

  public getMeterClass(value: number): string {
    if (value < 50) return 'low';
    if (value < 80) return 'medium';
    return 'high';
  }

  public formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  }

  // These methods are now handled by the main application directly
}