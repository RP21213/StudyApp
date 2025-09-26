/**
 * Loading Spinner Module
 * Provides reusable loading spinner functionality to prevent multiple clicks and show loading states
 */

class LoadingSpinner {
    constructor(buttonId, options = {}) {
        this.buttonId = buttonId;
        this.button = document.getElementById(buttonId);
        this.options = {
            loadingText: 'Loading...',
            successText: 'Success!',
            errorText: 'Try Again',
            successIcon: 'fas fa-check',
            errorIcon: 'fas fa-exclamation-triangle',
            originalIcon: 'fas fa-arrow-right',
            successDelay: 1000,
            ...options
        };
        
        if (!this.button) {
            console.error(`Button with ID '${buttonId}' not found`);
            return;
        }
        
        this.init();
    }
    
    init() {
        // Create spinner element if it doesn't exist
        if (!this.button.querySelector('.loading-spinner')) {
            const spinner = document.createElement('div');
            spinner.className = 'loading-spinner hidden';
            this.button.appendChild(spinner);
        }
        
        // Wrap button text in a span if it doesn't exist
        const textElements = this.button.querySelectorAll('*');
        let hasTextSpan = false;
        textElements.forEach(el => {
            if (el.textContent && !el.classList.contains('loading-spinner') && !el.classList.contains('fas')) {
                hasTextSpan = true;
            }
        });
        
        if (!hasTextSpan) {
            const textSpan = document.createElement('span');
            textSpan.className = 'btn-text';
            textSpan.textContent = this.button.textContent.trim();
            this.button.innerHTML = '';
            this.button.appendChild(textSpan);
        }
    }
    
    showLoading(text = null) {
        if (this.button.classList.contains('button-loading')) {
            return false; // Already loading
        }
        
        this.button.classList.add('button-loading');
        
        const textSpan = this.button.querySelector('.btn-text') || this.button;
        const icon = this.button.querySelector('i');
        const spinner = this.button.querySelector('.loading-spinner');
        
        // Update text
        textSpan.textContent = text || this.options.loadingText;
        
        // Hide icon and show spinner
        if (icon) icon.classList.add('hidden');
        if (spinner) spinner.classList.remove('hidden');
        
        return true;
    }
    
    showSuccess(text = null, callback = null) {
        const textSpan = this.button.querySelector('.btn-text') || this.button;
        const icon = this.button.querySelector('i');
        const spinner = this.button.querySelector('.loading-spinner');
        
        // Update text
        textSpan.textContent = text || this.options.successText;
        
        // Hide spinner and show success icon
        if (spinner) spinner.classList.add('hidden');
        if (icon) {
            icon.classList.remove('hidden');
            icon.className = this.options.successIcon + ' ml-2';
        }
        
        // Execute callback after delay
        if (callback) {
            setTimeout(callback, this.options.successDelay);
        }
    }
    
    showError(text = null) {
        this.reset();
        
        const textSpan = this.button.querySelector('.btn-text') || this.button;
        const icon = this.button.querySelector('i');
        
        // Update text
        textSpan.textContent = text || this.options.errorText;
        
        // Show error icon
        if (icon) {
            icon.classList.remove('hidden');
            icon.className = this.options.errorIcon + ' ml-2';
        }
        
        // Auto-reset after 3 seconds
        setTimeout(() => this.reset(), 3000);
    }
    
    reset() {
        this.button.classList.remove('button-loading');
        
        const textSpan = this.button.querySelector('.btn-text') || this.button;
        const icon = this.button.querySelector('i');
        const spinner = this.button.querySelector('.loading-spinner');
        
        // Reset text to original
        const originalText = this.button.getAttribute('data-original-text') || 'Explore Now';
        textSpan.textContent = originalText;
        
        // Show original icon and hide spinner
        if (icon) {
            icon.classList.remove('hidden');
            icon.className = this.options.originalIcon + ' ml-2';
        }
        if (spinner) spinner.classList.add('hidden');
    }
    
    isLoading() {
        return this.button.classList.contains('button-loading');
    }
}

// Utility function to create a loading spinner for any button
function createLoadingSpinner(buttonId, options = {}) {
    return new LoadingSpinner(buttonId, options);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LoadingSpinner, createLoadingSpinner };
}
