/**
 * LiveFeed Component - Single camera feed with real-time tracking
 *
 * Features:
 * - MJPEG video stream from Node.js (proxied from Python)
 * - Real-time tracking data via WebSocket
 * - Connection status indicator
 * - Object count and statistics
 *
 * Props:
 *   - cameraId: Camera identifier (e.g., 'camera_1')
 *   - showStats: Show tracking statistics (default: true)
 */

import { useTracking } from '../hooks/useTracking';

export default function LiveFeed({ cameraId, showStats = true, className = '' }) {
  const { trackingData, connected, error } = useTracking(cameraId);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
  const streamUrl = `${BACKEND_URL}/api/stream/${cameraId}/stream`;

  const objectCount = trackingData?.objects?.length || 0;
  const lastUpdate = trackingData?.timestamp
    ? new Date(trackingData.timestamp * 1000).toLocaleTimeString()
    : '--';

  // Calculate highest risk level
  const highestRisk = trackingData?.objects?.reduce((max, obj) => {
    return (obj.risk_score || 0) > max ? (obj.risk_score || 0) : max;
  }, 0) || 0;

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'critical': return 'bg-red-600 text-white';
      case 'high': return 'bg-red-500 text-white';
      case 'medium': return 'bg-orange-500 text-white';
      case 'low': return 'bg-yellow-500 text-white';
      default: return 'bg-green-500 text-white';
    }
  };

  const getRiskBadgeColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'critical': return 'bg-red-600';
      case 'high': return 'bg-red-500';
      case 'medium': return 'bg-orange-500';
      case 'low': return 'bg-yellow-500';
      default: return 'bg-green-500';
    }
  };

  return (
    <div className={`bg-white rounded-lg shadow-lg overflow-hidden ${className}`}>
      {/* Camera Header */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-700 px-4 py-3 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="text-white font-semibold text-lg">
            📹 {cameraId.replace('_', ' ').toUpperCase()}
          </div>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
          <span className={`text-sm font-medium ${connected ? 'text-green-300' : 'text-red-300'}`}>
            {connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Video Stream */}
      <div className="relative bg-black aspect-video">
        <img
          src={streamUrl}
          alt={`Camera ${cameraId}`}
          className="w-full h-full object-contain"
          onError={(e) => {
            e.target.style.display = 'none';
            e.target.nextElementSibling.style.display = 'flex';
          }}
        />

        {/* Error/Loading Overlay */}
        <div className="absolute inset-0 hidden flex-col items-center justify-center bg-gray-900 text-white">
          <div className="text-4xl mb-4">📹</div>
          <div className="text-sm text-gray-400">Stream unavailable</div>
        </div>

        {/* Live Indicator */}
        {connected && (
          <div className="absolute top-3 left-3 bg-red-600 text-white px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2 shadow-lg">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            LIVE
          </div>
        )}

        {/* Object Count & Risk Badge */}
        {showStats && objectCount > 0 && (
          <div className="absolute top-3 right-3 flex flex-col gap-2 items-end">
            <div className="bg-blue-600 bg-opacity-90 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg">
              {objectCount} {objectCount === 1 ? 'Person' : 'People'}
            </div>
            {highestRisk > 0 && (
              <div className={`${getRiskBadgeColor(trackingData.objects.find(o => o.risk_score === highestRisk)?.risk_level)} bg-opacity-90 text-white px-4 py-2 rounded-lg text-xs font-bold shadow-lg animate-pulse`}>
                ⚠️ Risk: {highestRisk}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Tracking Statistics */}
      {showStats && (
        <div className="p-4 bg-gray-50 border-t border-gray-200">
          <div className="grid grid-cols-3 gap-4">
            {/* People Detected */}
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{objectCount}</div>
              <div className="text-xs text-gray-600 mt-1">Detected</div>
            </div>

            {/* Last Update */}
            <div className="text-center border-x border-gray-300">
              <div className="text-sm font-semibold text-gray-800">{lastUpdate}</div>
              <div className="text-xs text-gray-600 mt-1">Last Update</div>
            </div>

            {/* Status */}
            <div className="text-center">
              <div className={`text-sm font-semibold ${connected ? 'text-green-600' : 'text-red-600'}`}>
                {connected ? '✓ Active' : '✗ Inactive'}
              </div>
              <div className="text-xs text-gray-600 mt-1">Status</div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
              ⚠️ {error}
            </div>
          )}

          {/* Tracking Details */}
          {trackingData?.objects && trackingData.objects.length > 0 && (
            <details className="mt-3">
              <summary className="text-xs text-blue-600 cursor-pointer hover:text-blue-700 font-medium">
                View tracking details ({trackingData.objects.length} objects)
              </summary>
              <div className="mt-2 max-h-48 overflow-y-auto space-y-2">
                {trackingData.objects.map((obj, idx) => (
                  <div key={idx} className={`text-xs p-3 rounded-lg border-2 ${getRiskColor(obj.risk_level)} bg-opacity-10 border-opacity-30`}>
                    {/* Header - Track ID and Risk */}
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-bold text-gray-800">
                        🎯 Track #{obj.track_id || idx + 1}
                      </span>
                      {obj.risk_level && (
                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${getRiskColor(obj.risk_level)}`}>
                          {obj.risk_level.toUpperCase()} ({obj.risk_score || 0})
                        </span>
                      )}
                    </div>

                    {/* Feature Grid */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {/* Speed */}
                      {obj.speed !== undefined && (
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600">⚡ Speed:</span>
                          <span className="font-semibold text-gray-800">{obj.speed.toFixed(1)} px/s</span>
                        </div>
                      )}

                      {/* Edge Distance */}
                      {obj.dist_to_edge !== undefined && (
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600">📏 Edge:</span>
                          <span className="font-semibold text-gray-800">{obj.dist_to_edge.toFixed(1)} px</span>
                        </div>
                      )}

                      {/* Dwell Time */}
                      {obj.dwell_time !== undefined && (
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600">⏱️ Dwell:</span>
                          <span className="font-semibold text-gray-800">{obj.dwell_time.toFixed(1)}s</span>
                        </div>
                      )}

                      {/* Torso Angle */}
                      {obj.torso_angle !== undefined && obj.torso_angle !== null && (
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600">🔄 Angle:</span>
                          <span className="font-semibold text-gray-800">{obj.torso_angle.toFixed(0)}°</span>
                        </div>
                      )}

                      {/* Confidence */}
                      <div className="flex items-center gap-1">
                        <span className="text-gray-600">✓ Conf:</span>
                        <span className="font-semibold text-gray-800">{(obj.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
