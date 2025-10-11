# Spaced Repetition System Fix

## Problem
The spaced repetition system was showing cards as perpetually due with:
- `repetitions: 0`
- `interval: 1d`

This indicated that card reviews were not being persisted to the database.

## Root Causes Identified

### 1. **Document ID in Update Dictionary**
When calling `card_ref.update(sr_card.to_dict())`, the dictionary included the `id` field. While Firestore typically ignores this, it could potentially cause issues with the update operation.

**Fix:** Removed the `id` field from the update dictionary before calling `update()`.

### 2. **Type Coercion Issues**
Numeric fields (`ease_factor`, `interval_days`, `repetitions`) might not have been properly typed when serializing to Firestore.

**Fix:** Explicitly cast these fields to `float` and `int` in the `to_dict()` method.

### 3. **Timezone-Aware Datetime Handling**
The `from_dict()` method wasn't ensuring datetime objects from Firestore were timezone-aware.

**Fix:** Added explicit timezone handling in `from_dict()` to ensure all datetime objects have UTC timezone information.

### 4. **Insufficient Error Logging**
Update failures might have been occurring silently without proper logging.

**Fix:** Added comprehensive error handling and verification logging to the review endpoint.

## Changes Made

### `models.py` - SpacedRepetitionCard class

1. **Enhanced `to_dict()` method:**
```python
def to_dict(self):
    return {
        'id': self.id,
        'activity_id': self.activity_id,
        'card_index': self.card_index,
        'front': self.front,
        'back': self.back,
        'ease_factor': float(self.ease_factor),  # Explicit type conversion
        'interval_days': int(self.interval_days),
        'repetitions': int(self.repetitions),
        'last_reviewed': self.last_reviewed,
        'next_review': self.next_review,
        'difficulty': self.difficulty,
        'created_at': self.created_at,
    }
```

2. **Improved `from_dict()` method:**
   - Explicit field extraction using `.get()`
   - Timezone-aware datetime handling
   - Default values for all fields

### `app.py` - Review endpoint

1. **Enhanced update operation:**
   - Removes `id` field before update
   - Wraps update in try-catch with detailed logging
   - Verifies update persisted by re-reading the document
   - Logs before and after states

2. **Improved debugging:**
   - More detailed logging of card state before and after review
   - Verification step to confirm database changes
   - Better error messages with full tracebacks

3. **Enhanced due cards logging:**
   - Now logs `last_reviewed` and `next_review` timestamps
   - More comprehensive card state information

## How to Test

### Option 1: Use the Test Script
```bash
python test_sr_update.py
```

This will:
1. Find a test card in your database
2. Update it with new values
3. Verify the update persisted
4. Report success or failure

### Option 2: Manual Testing
1. Open your application
2. Navigate to a hub with flashcards
3. Start a spaced repetition review session
4. Review some cards (rate them as "Good", "Easy", etc.)
5. Check the server logs for debug messages:
   - Look for "✅ SUCCESS: Card {id} updated in Firestore"
   - Look for "🔍 VERIFY: Card after update"
6. Refresh the dashboard
7. Check if the due card count decreases

### Option 3: Check Server Logs
After reviewing cards, your logs should now show:

```
🔍 DEBUG: Card state after review: interval=6d, ease=2.50, reps=1
🔍 DEBUG: About to update card {id} with data: repetitions=1, interval=6d, next_review={timestamp}
✅ SUCCESS: Card {id} updated in Firestore: 1d -> 6d
🔍 VERIFY: Card after update - repetitions=1, interval=6d
```

If you see a "⚠️ WARNING" or "❌ ERROR" message, that indicates the specific issue.

## Expected Behavior After Fix

1. **First Review (rating: "Good" or better):**
   - `repetitions`: 0 → 1
   - `interval_days`: 1 → 1
   - `last_reviewed`: None → current timestamp
   - `next_review`: updated to 1 day from now

2. **Second Review (rating: "Good" or better):**
   - `repetitions`: 1 → 2
   - `interval_days`: 1 → 6
   - `next_review`: updated to 6 days from now

3. **Third Review (rating: "Good"):**
   - `repetitions`: 2 → 3
   - `interval_days`: 6 → 15 (6 × 2.5 ease factor)
   - `next_review`: updated to 15 days from now

4. **If rated "Again" or "Hard":**
   - `repetitions`: reset to 0
   - `interval_days`: reset to 1
   - Card becomes due again immediately or after 1 day

## Monitoring

Check your logs regularly for:
- **Success indicators:** ✅ SUCCESS messages
- **Warnings:** ⚠️ WARNING messages (mismatch in expected vs actual values)
- **Errors:** ❌ ERROR messages (update failures)

## If Issues Persist

If cards are still not updating properly:

1. Check for ❌ ERROR messages in the logs
2. Verify Firestore permissions allow updates
3. Check if there are Firestore index issues
4. Run the test script to isolate the problem
5. Check browser console for JavaScript errors preventing API calls

## Additional Notes

- All changes are backward compatible
- No database migration required
- Existing cards will work with the new code
- The fix includes comprehensive logging for future debugging

