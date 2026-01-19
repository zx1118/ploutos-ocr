-- ============================================================
-- OCR 清理积压 PENDING 任务
-- 执行前请确认这些任务确实是需要清理的测试数据
-- ============================================================

-- 1. 查看当前 PENDING 任务数量
SELECT status, COUNT(*) as count FROM ocr_task GROUP BY status;

-- 2. 查看 PENDING 任务详情（确认是否可以删除）
SELECT id, file_name, create_time, create_by 
FROM ocr_task 
WHERE status = 'PENDING' 
ORDER BY create_time DESC;

-- ============================================================
-- 方案A: 删除所有 PENDING 任务（适用于测试数据）
-- ============================================================
-- DELETE FROM ocr_task WHERE status = 'PENDING';

-- ============================================================
-- 方案B: 将 PENDING 任务标记为 CANCELLED（保留记录）
-- ============================================================
-- UPDATE ocr_task SET status = 'CANCELLED', error_message = '手动清理积压任务', update_time = NOW() WHERE status = 'PENDING';

-- ============================================================
-- 方案C: 只清理超过1小时的 PENDING 任务
-- ============================================================
-- DELETE FROM ocr_task WHERE status = 'PENDING' AND create_time < DATE_SUB(NOW(), INTERVAL 1 HOUR);

-- ============================================================
-- 清理 Redis Stream 中的积压消息（在命令行执行）
-- ============================================================
-- redis-cli XTRIM ocr:tasks MAXLEN 0
-- 或者保留最近100条：
-- redis-cli XTRIM ocr:tasks MAXLEN 100

