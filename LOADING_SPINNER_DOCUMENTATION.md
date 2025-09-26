# Loading Spinner Module Documentation

## Overview
The Loading Spinner module provides a reusable solution to prevent multiple clicks on buttons and show loading states during asynchronous operations. This is particularly useful for preventing users from creating multiple study hubs by clicking the "Explore Now" button multiple times.

## Files Created/Modified

### 1. `/static/js/loading-spinner.js`
- **Purpose**: Reusable JavaScript class for loading spinner functionality
- **Features**: 
  - Prevents multiple clicks during loading
  - Customizable text and icons
  - Success and error state handling
  - Automatic button state management

### 2. `/templates/onboarding.html` (Modified)
- **Purpose**: Updated the "Explore Now" button to use the loading spinner
- **Changes**:
  - Added loading spinner CSS styles
  - Updated button HTML structure
  - Modified `completeOnboarding()` function to use LoadingSpinner class
  - Added click prevention logic

### 3. `/templates/loading_spinner_test.html` (New)
- **Purpose**: Test page to demonstrate loading spinner functionality
- **Access**: Visit `/test/loading-spinner` route
- **Features**: Three test buttons showing different scenarios

### 4. `/app.py` (Modified)
- **Purpose**: Added test route for loading spinner
- **New Route**: `/test/loading-spinner`

## Usage Instructions

### Basic Usage
```javascript
// Create a loading spinner instance
const spinner = new LoadingSpinner('button-id', {
    loadingText: 'Loading...',
    successText: 'Success!',
    errorText: 'Try Again',
    successDelay: 1000
});

// Show loading state
spinner.showLoading();

// Show success state
spinner.showSuccess('Operation Complete!', () => {
    // Callback function after success delay
    window.location.href = '/dashboard';
});

// Show error state
spinner.showError('Something went wrong');
```

### Advanced Configuration
```javascript
const spinner = new LoadingSpinner('my-button', {
    loadingText: 'Creating Your Study Hub...',
    successText: 'Redirecting...',
    errorText: 'Error - Try Again',
    successIcon: 'fas fa-check',
    errorIcon: 'fas fa-exclamation-triangle',
    originalIcon: 'fas fa-arrow-right',
    successDelay: 1500
});
```

## CSS Requirements

Add these styles to your CSS:

```css
/* Loading Spinner Styles */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.button-loading {
    opacity: 0.7;
    cursor: not-allowed;
    pointer-events: none;
}

.button-loading .loading-spinner {
    margin-right: 8px;
}
```

## HTML Structure

Your button should have this structure:

```html
<button id="my-button" class="your-button-classes">
    <span class="btn-text">Button Text</span>
    <i class="fas fa-arrow-right ml-2"></i>
</button>
```

## API Reference

### LoadingSpinner Class

#### Constructor
```javascript
new LoadingSpinner(buttonId, options)
```
- `buttonId`: ID of the button element
- `options`: Configuration object (optional)

#### Methods

##### `showLoading(text)`
- Shows loading state
- Returns `false` if already loading (prevents multiple clicks)
- `text`: Custom loading text (optional)

##### `showSuccess(text, callback)`
- Shows success state
- `text`: Custom success text (optional)
- `callback`: Function to execute after success delay (optional)

##### `showError(text)`
- Shows error state and auto-resets after 3 seconds
- `text`: Custom error text (optional)

##### `reset()`
- Resets button to original state

##### `isLoading()`
- Returns `true` if button is currently in loading state

## Testing

1. **Test Page**: Visit `/test/loading-spinner` to see the loading spinner in action
2. **Onboarding**: The "Explore Now" button now has loading spinner functionality
3. **Multiple Clicks**: Try clicking buttons multiple times - only the first click will be processed

## Benefits

1. **Prevents Multiple Submissions**: Users can't accidentally create multiple study hubs
2. **Better UX**: Clear visual feedback during operations
3. **Reusable**: Can be used on any button throughout the application
4. **Customizable**: Flexible configuration options
5. **Error Handling**: Built-in error state management

## Integration Examples

### With Fetch API
```javascript
const spinner = new LoadingSpinner('submit-btn');

document.getElementById('submit-btn').addEventListener('click', async function() {
    if (!spinner.showLoading()) return;
    
    try {
        const response = await fetch('/api/submit', {
            method: 'POST',
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            spinner.showSuccess('Success!', () => {
                window.location.href = '/success';
            });
        } else {
            spinner.showError('Error occurred');
        }
    } catch (error) {
        spinner.showError('Network error');
    }
});
```

### With Form Submission
```javascript
const spinner = new LoadingSpinner('form-submit-btn');

document.getElementById('my-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    if (!spinner.showLoading()) return;
    
    // Process form data
    const formData = new FormData(this);
    
    fetch('/api/form-submit', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            spinner.showSuccess('Form submitted!');
        } else {
            spinner.showError('Submission failed');
        }
    })
    .catch(error => {
        spinner.showError('Error occurred');
    });
});
```
