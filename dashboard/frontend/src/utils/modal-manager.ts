export class ModalManager {
  private static instance: ModalManager
  private activeModals = new Set<string>()
  private autoCloseTimer = new Map<string, number>()

  static getInstance(): ModalManager {
    if (!ModalManager.instance) {
      ModalManager.instance = new ModalManager()
    }
    return ModalManager.instance
  }

  showErrorModal(id: string, message: string, autoClose = true): void {
    const modal = document.getElementById(id)
    if (!modal) return

    this.activeModals.add(id)
    modal.setAttribute('data-modal', 'visible')

    if (autoClose) {
      // Auto-close error modals after 5 seconds
      const timer = window.setTimeout(() => {
        this.closeModal(id)
      }, 5000)
      this.autoCloseTimer.set(id, timer)
    }
  }

  closeModal(id: string): void {
    const modal = document.getElementById(id)
    if (!modal) return

    modal.setAttribute('data-modal', 'hidden')
    this.activeModals.delete(id)

    const timer = this.autoCloseTimer.get(id)
    if (timer) {
      clearTimeout(timer)
      this.autoCloseTimer.delete(id)
    }
  }

  closeAllModals(): void {
    this.activeModals.forEach((id) => this.closeModal(id))
  }
}
