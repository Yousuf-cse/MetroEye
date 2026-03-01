/**
 * Quick MongoDB Connection and Alert Test
 * Run with: node test-mongodb.js
 */

const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// Import Alert model
const Alert = require('./models/Alert');

async function testMongoDB() {
  console.log('\n🧪 Testing MongoDB Connection and Alert Saving\n');
  console.log('=' .repeat(60));

  try {
    // Step 1: Connect to MongoDB
    console.log('\n1️⃣  Connecting to MongoDB...');
    console.log('   URI:', process.env.MONGODB_URI);

    await mongoose.connect(process.env.MONGODB_URI);

    console.log('   ✅ Connected successfully!');
    console.log('   📂 Database:', mongoose.connection.name);
    console.log('   🔗 Host:', mongoose.connection.host);
    console.log('   🔌 Port:', mongoose.connection.port);

    // Step 2: Create test alert
    console.log('\n2️⃣  Creating test alert...');

    const testAlertData = {
      alert_id: `TEST-${uuidv4()}`,
      timestamp: new Date(),
      camera_id: 'test_camera',
      track_id: 999,
      risk_level: 'medium',
      confidence: 0.8,
      risk_score: 65,
      llm_reasoning: 'Test alert for debugging',
      alert_message: 'This is a test alert to verify MongoDB is working',
      recommended_action: 'monitor',
      status: 'pending'
    };

    console.log('   Alert data:', JSON.stringify(testAlertData, null, 2));

    const testAlert = new Alert(testAlertData);
    await testAlert.save();

    console.log('   ✅ Alert saved successfully!');
    console.log('   MongoDB ID:', testAlert._id);

    // Step 3: Query the alert back
    console.log('\n3️⃣  Querying saved alert...');

    const foundAlert = await Alert.findOne({ alert_id: testAlertData.alert_id });

    if (foundAlert) {
      console.log('   ✅ Alert found in database!');
      console.log('   Alert ID:', foundAlert.alert_id);
      console.log('   Track ID:', foundAlert.track_id);
      console.log('   Risk Score:', foundAlert.risk_score);
    } else {
      console.log('   ❌ Alert NOT found in database!');
    }

    // Step 4: Count total alerts
    console.log('\n4️⃣  Checking all alerts in database...');

    const totalAlerts = await Alert.countDocuments();
    console.log('   Total alerts in DB:', totalAlerts);

    if (totalAlerts > 0) {
      console.log('\n   Recent alerts:');
      const recentAlerts = await Alert.find().sort({ timestamp: -1 }).limit(5);
      recentAlerts.forEach((alert, i) => {
        console.log(`   ${i + 1}. ${alert.alert_id} - Track ${alert.track_id} - Risk ${alert.risk_score}`);
      });
    }

    // Step 5: Clean up test alert
    console.log('\n5️⃣  Cleaning up test alert...');
    await Alert.deleteOne({ alert_id: testAlertData.alert_id });
    console.log('   ✅ Test alert deleted');

    console.log('\n' + '='.repeat(60));
    console.log('✅ ALL TESTS PASSED! MongoDB is working correctly.\n');

  } catch (error) {
    console.error('\n❌ TEST FAILED!');
    console.error('Error:', error.message);

    if (error.errors) {
      console.error('\nValidation errors:');
      Object.keys(error.errors).forEach(key => {
        console.error(`  - ${key}: ${error.errors[key].message}`);
      });
    }

    console.error('\nFull error:', error);
    console.log('\n' + '='.repeat(60));

  } finally {
    // Close connection
    await mongoose.connection.close();
    console.log('🔌 MongoDB connection closed\n');
    process.exit(0);
  }
}

// Run the test
testMongoDB();
