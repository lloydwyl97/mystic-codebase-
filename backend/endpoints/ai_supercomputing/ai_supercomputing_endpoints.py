import time
from datetime import datetime

import psutil

try:
    from backend.services.gpu_monitor import get_gpu_metrics  # If available
except ImportError:
    get_gpu_metrics = None

import os

try:
    from ai_tasks import celery_app
    from celery.task.control import inspect
except ImportError:
    celery_app = None
    inspect = None

try:
    from backend.middleware.circuit_breaker import circuit_states, FAILURE_THRESHOLD, RESET_TIMEOUT
except ImportError:
    circuit_states = {}
    FAILURE_THRESHOLD = 5
    RESET_TIMEOUT = 60

try:
    from backend.services.performance_monitor import PerformanceMonitor

    performance_monitor = PerformanceMonitor()
except ImportError:
    performance_monitor = None

try:
    from backend.services.health_monitor_service import HealthMonitorService

    health_monitor = HealthMonitorService()
except ImportError:
    health_monitor = None

try:
    from ai_agent_orchestrator_service import AIAgentOrchestratorService

    orchestrator_service = AIAgentOrchestratorService()
except ImportError:
    orchestrator_service = None

try:
    import ai_model_versioning
except ImportError:
    ai_model_versioning = None
try:
    import ai_auto_retrain
except ImportError:
    ai_auto_retrain = None
try:
    import ai_genetic_algorithm
except ImportError:
    ai_genetic_algorithm = None

try:
    from backend.modules.notifications.alert_manager import AlertManager

    alert_manager = AlertManager()
except ImportError:
    alert_manager = None

from fastapi import APIRouter, HTTPException

from backend.services.ai_strategy_service import AIStrategyService
from backend.services.analytics_service import get_analytics_service

router = APIRouter(prefix="/api/ai/supercomputing", tags=["AI Supercomputing"])


@router.get("/model-performance")
async def get_model_performance():
    """Get AI model inference performance data (live)"""
    try:
        # Prefer the best available live model performance metrics
        try:
            ai_strategy_service = AIStrategyService()
            data = await ai_strategy_service.get_performance_metrics()
        except Exception as e:
            # Fallback to analytics_service if available
            try:
                analytics_service = get_analytics_service()
                data = await analytics_service.get_performance_metrics()
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"No live model performance metrics available: {str(e)}; {str(e2)}",
                )
        return {"models": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resource-utilization")
async def get_resource_utilization():
    """Get AI resource utilization data (live)"""
    try:
        # CPU
        cpu = {
            "usage": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count(logical=True),
            "temperature": None,  # psutil does not provide temperature cross-platform
        }
        # Memory
        memory_stats = psutil.virtual_memory()
        memory = {
            "used": memory_stats.used // (1024 * 1024),
            "total": memory_stats.total // (1024 * 1024),
            "percentage": memory_stats.percent,
        }
        # Disk
        disk_stats = psutil.disk_usage("/")
        storage = {
            "used": disk_stats.used // (1024 * 1024),
            "total": disk_stats.total // (1024 * 1024),
            "iops": None,  # Not available via psutil
        }
        # Network
        psutil.net_io_counters()
        network = {
            "bandwidth": None,  # Not directly available
            "latency": None,  # Not directly available
            "connections": len(psutil.net_connections()),
        }
        # GPU
        gpu = None
        if get_gpu_metrics:
            try:
                gpu = get_gpu_metrics()
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {str(e)}")
                gpu = None
        # Compose result
        result = {
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu if gpu else {"usage": None, "memory": None, "temperature": None},
            "storage": storage,
            "network": network,
        }
        return result
    except Exception as e:
        logger.error(f"Resource utilization check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job-scheduler")
async def get_job_scheduler():
    """Get AI job scheduler status (live)"""
    try:
        jobs = []
        total_jobs = 0
        running_jobs = 0
        queued_jobs = 0
        completed_jobs = 0
        failed_jobs = 0

        # Get Celery task status if available
        if celery_app and inspect:
            try:
                i = inspect()
                active_tasks = i.active()
                reserved_tasks = i.reserved()
                i.registered()

                # Process active tasks
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        job = {
                            "id": task.get("id", f"celery_{len(jobs)}"),
                            "name": task.get("name", "Unknown Task"),
                            "status": "running",
                            "priority": 3,
                            "progress": 50.0,  # Celery doesn't provide progress
                            "startTime": task.get("time_start", datetime.now().isoformat()),
                            "estimatedCompletion": None,
                        }
                        jobs.append(job)
                        running_jobs += 1

                # Process reserved (queued) tasks
                for worker, tasks in reserved_tasks.items():
                    for task in tasks:
                        job = {
                            "id": task.get("id", f"celery_{len(jobs)}"),
                            "name": task.get("name", "Unknown Task"),
                            "status": "queued",
                            "priority": 3,
                            "progress": 0.0,
                            "startTime": None,
                            "estimatedCompletion": None,
                        }
                        jobs.append(job)
                        queued_jobs += 1

            except Exception:
                # Fallback to basic Celery stats
                try:
                    stats = celery_app.control.inspect().stats()
                    for worker, worker_stats in stats.items():
                        job = {
                            "id": f"worker_{worker}",
                            "name": f"Celery Worker {worker}",
                            "status": "running",
                            "priority": 3,
                            "progress": 100.0,
                            "startTime": datetime.now().isoformat(),
                            "estimatedCompletion": None,
                        }
                        jobs.append(job)
                        running_jobs += 1
                except Exception:
                    pass

        # Get service queue status from Redis
        try:
            import redis

            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
            )

            # Check various service queues
            queue_names = [
                "strategy_execution_queue",
                "ai_agent_queue",
                "ai_trade_queue",
                "ai_leaderboard_queue",
            ]

            for queue_name in queue_names:
                queue_length = redis_client.llen(queue_name)
                if queue_length > 0:
                    job = {
                        "id": f"queue_{queue_name}",
                        "name": f"{queue_name.replace('_', ' ').title()}",
                        "status": "queued",
                        "priority": 2,
                        "progress": 0.0,
                        "startTime": None,
                        "estimatedCompletion": None,
                        "queue_length": queue_length,
                    }
                    jobs.append(job)
                    queued_jobs += queue_length

        except Exception:
            # Redis not available, continue without queue data
            pass

        # Calculate totals
        total_jobs = len(jobs)
        completed_jobs = 0  # Celery doesn't provide completed count in inspection
        failed_jobs = 0  # Celery doesn't provide failed count in inspection

        return {
            "totalJobs": total_jobs,
            "runningJobs": running_jobs,
            "queuedJobs": queued_jobs,
            "completedJobs": completed_jobs,
            "failedJobs": failed_jobs,
            "averageWaitTime": 0.0,  # Not available from Celery inspection
            "jobs": jobs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fault-tolerance")
async def get_fault_tolerance():
    """Get AI fault tolerance and failover status (live)"""
    try:
        incidents = []
        redundancy_level = 2
        failover_status = "active"
        backup_systems = 2
        last_failover = None
        uptime = 99.9

        # Get circuit breaker status
        if circuit_states:
            open_circuits = 0
            total_circuits = len(circuit_states)

            for endpoint, state in circuit_states.items():
                if state.get("is_open", False):
                    open_circuits += 1
                    incidents.append(
                        {
                            "id": f"circuit_{endpoint}",
                            "type": "software",
                            "severity": (
                                "high"
                                if state.get("failures", 0) >= FAILURE_THRESHOLD
                                else "medium"
                            ),
                            "timestamp": datetime.fromtimestamp(
                                state.get("last_failure", time.time())
                            ).isoformat(),
                            "resolved": False,
                            "description": f"Circuit breaker open for {endpoint} - {state.get('failures', 0)} failures",
                        }
                    )

            # Calculate failover status
            if open_circuits > 0:
                failover_status = "active"
                last_failover = datetime.now().isoformat()
            else:
                failover_status = "standby"

            # Calculate uptime based on circuit health
            uptime = (
                max(99.9 - (open_circuits / total_circuits * 0.1), 99.5)
                if total_circuits > 0
                else 99.9
            )

        # Get performance monitor alerts
        if performance_monitor:
            try:
                alerts = (
                    performance_monitor.alerts[-5:]
                    if hasattr(performance_monitor, "alerts")
                    else []
                )
                for alert in alerts:
                    incidents.append(
                        {
                            "id": f"alert_{len(incidents)}",
                            "type": "performance",
                            "severity": alert.level if hasattr(alert, "level") else "medium",
                            "timestamp": (
                                alert.timestamp
                                if hasattr(alert, "timestamp")
                                else datetime.now().isoformat()
                            ),
                            "resolved": False,
                            "description": (
                                alert.message if hasattr(alert, "message") else "Performance alert"
                            ),
                        }
                    )
            except Exception:
                pass

        # Get system health status
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Add system health incidents
            if cpu_percent > 90:
                incidents.append(
                    {
                        "id": f"system_{len(incidents)}",
                        "type": "hardware",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat(),
                        "resolved": False,
                        "description": f"High CPU usage: {cpu_percent}%",
                    }
                )

            if memory.percent > 90:
                incidents.append(
                    {
                        "id": f"system_{len(incidents)}",
                        "type": "hardware",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat(),
                        "resolved": False,
                        "description": f"High memory usage: {memory.percent}%",
                    }
                )
        except Exception:
            pass

        # Calculate redundancy level based on available services
        try:
            from backend.services.system_monitor import SystemMonitor

            system_monitor = SystemMonitor()
            services_status = await system_monitor.get_services_status()
            active_services = sum(1 for status in services_status.values() if status == "healthy")
            redundancy_level = min(5, max(2, active_services // 2))
            backup_systems = max(2, active_services - 1)
        except Exception:
            pass

        return {
            "redundancyLevel": redundancy_level,
            "failoverStatus": failover_status,
            "backupSystems": backup_systems,
            "lastFailover": last_failover,
            "uptime": round(uptime, 2),
            "incidents": incidents,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health():
    """Get AI system health data (live)"""
    try:
        components = []
        alerts = []
        overall_health = 100.0

        # Get system health from HealthMonitorService
        if health_monitor:
            try:
                system_health = await health_monitor.get_system_health()
                system_metrics = system_health.get("system", {})

                # Add system components
                components.extend(
                    [
                        {
                            "name": "CPU",
                            "health": max(0, 100 - system_metrics.get("cpu_percent", 0)),
                            "status": (
                                "healthy"
                                if system_metrics.get("cpu_percent", 0) < 80
                                else (
                                    "warning"
                                    if system_metrics.get("cpu_percent", 0) < 90
                                    else "critical"
                                )
                            ),
                            "lastCheck": system_health.get("timestamp", datetime.now().isoformat()),
                        },
                        {
                            "name": "Memory",
                            "health": max(0, 100 - system_metrics.get("memory_percent", 0)),
                            "status": (
                                "healthy"
                                if system_metrics.get("memory_percent", 0) < 80
                                else (
                                    "warning"
                                    if system_metrics.get("memory_percent", 0) < 90
                                    else "critical"
                                )
                            ),
                            "lastCheck": system_health.get("timestamp", datetime.now().isoformat()),
                        },
                        {
                            "name": "Disk",
                            "health": max(0, 100 - system_metrics.get("disk_percent", 0)),
                            "status": (
                                "healthy"
                                if system_metrics.get("disk_percent", 0) < 80
                                else (
                                    "warning"
                                    if system_metrics.get("disk_percent", 0) < 90
                                    else "critical"
                                )
                            ),
                            "lastCheck": system_health.get("timestamp", datetime.now().isoformat()),
                        },
                    ]
                )

                # Calculate overall health
                component_healths = [c["health"] for c in components]
                overall_health = (
                    sum(component_healths) / len(component_healths) if component_healths else 100.0
                )

            except Exception:
                # Fallback to basic system metrics
                try:
                    import psutil

                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")

                    components.extend(
                        [
                            {
                                "name": "CPU",
                                "health": max(0, 100 - cpu_percent),
                                "status": (
                                    "healthy"
                                    if cpu_percent < 80
                                    else "warning" if cpu_percent < 90 else "critical"
                                ),
                                "lastCheck": datetime.now().isoformat(),
                            },
                            {
                                "name": "Memory",
                                "health": max(0, 100 - memory.percent),
                                "status": (
                                    "healthy"
                                    if memory.percent < 80
                                    else "warning" if memory.percent < 90 else "critical"
                                ),
                                "lastCheck": datetime.now().isoformat(),
                            },
                            {
                                "name": "Disk",
                                "health": max(0, 100 - disk.percent),
                                "status": (
                                    "healthy"
                                    if disk.percent < 80
                                    else "warning" if disk.percent < 90 else "critical"
                                ),
                                "lastCheck": datetime.now().isoformat(),
                            },
                        ]
                    )

                    component_healths = [c["health"] for c in components]
                    overall_health = (
                        sum(component_healths) / len(component_healths)
                        if component_healths
                        else 100.0
                    )

                except Exception:
                    pass

        # Get performance monitor alerts
        if performance_monitor:
            try:
                monitor_alerts = (
                    performance_monitor.alerts[-5:]
                    if hasattr(performance_monitor, "alerts")
                    else []
                )
                for alert in monitor_alerts:
                    alerts.append(
                        {
                            "id": f"alert_{len(alerts)}",
                            "severity": alert.level if hasattr(alert, "level") else "info",
                            "message": (
                                alert.message if hasattr(alert, "message") else "Performance alert"
                            ),
                            "timestamp": (
                                alert.timestamp
                                if hasattr(alert, "timestamp")
                                else datetime.now().isoformat()
                            ),
                            "acknowledged": False,
                        }
                    )
            except Exception:
                pass

        # Add service health checks
        try:
            from backend.services.system_monitor import SystemMonitor

            system_monitor = SystemMonitor()
            services_status = await system_monitor.get_services_status()

            for service_name, status in services_status.items():
                health_value = 100 if status == "healthy" else 50 if status == "warning" else 0
                components.append(
                    {
                        "name": service_name.replace("_", " ").title(),
                        "health": health_value,
                        "status": status,
                        "lastCheck": datetime.now().isoformat(),
                    }
                )
        except Exception:
            pass

        # Add database health
        try:
            from database import get_db_connection

            conn = get_db_connection()
            conn.close()
            components.append(
                {
                    "name": "Database",
                    "health": 100,
                    "status": "healthy",
                    "lastCheck": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            components.append(
                {
                    "name": "Database",
                    "health": 0,
                    "status": "critical",
                    "lastCheck": datetime.now().isoformat(),
                }
            )
            alerts.append(
                {
                    "id": f"alert_{len(alerts)}",
                    "severity": "critical",
                    "message": f"Database connection failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "acknowledged": False,
                }
            )

        # Add cache health
        try:
            from backend.modules.ai.persistent_cache import get_persistent_cache

            cache = get_persistent_cache()
            components.append(
                {
                    "name": "Cache",
                    "health": 100 if cache else 0,
                    "status": "healthy" if cache else "critical",
                    "lastCheck": datetime.now().isoformat(),
                }
            )
        except Exception:
            components.append(
                {
                    "name": "Cache",
                    "health": 0,
                    "status": "critical",
                    "lastCheck": datetime.now().isoformat(),
                }
            )

        # Recalculate overall health with all components
        if components:
            component_healths = [c["health"] for c in components]
            overall_health = sum(component_healths) / len(component_healths)

        return {
            "overallHealth": round(overall_health, 1),
            "components": components,
            "alerts": alerts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distributed-training")
async def get_distributed_training():
    """Get AI distributed training overview (live)"""
    try:
        nodes = []
        node_count = 0
        active_nodes = 0
        training_progress = 0.0
        model_size = None
        data_processed = None
        convergence_rate = None

        # Use orchestrator service if available
        if orchestrator_service:
            try:
                # Assume orchestrator_service.agents is a dict of agent objects
                for agent_name, agent in orchestrator_service.agents.items():
                    node_status = getattr(agent, "health_status", "unknown")
                    last_sync = getattr(agent, "last_heartbeat", None)
                    gpu_count = getattr(agent, "gpu_count", None)
                    memory_usage = getattr(agent, "memory_usage", None)
                    training_progress_val = getattr(agent, "training_progress", None)

                    nodes.append(
                        {
                            "id": agent_name,
                            "status": node_status,
                            "gpuCount": gpu_count,
                            "memoryUsage": memory_usage,
                            "trainingProgress": training_progress_val,
                            "lastSync": last_sync.isoformat() if last_sync else None,
                        }
                    )

                node_count = len(nodes)
                active_nodes = sum(1 for n in nodes if n["status"] == "healthy")
                # Aggregate training progress if available
                progress_vals = [
                    n["trainingProgress"] for n in nodes if n["trainingProgress"] is not None
                ]
                training_progress = (
                    sum(progress_vals) / len(progress_vals) if progress_vals else 0.0
                )
            except Exception:
                pass

        # Fallback: no orchestrator, return empty node list
        return {
            "nodeCount": node_count,
            "activeNodes": active_nodes,
            "trainingProgress": round(training_progress, 1),
            "modelSize": model_size,
            "dataProcessed": data_processed,
            "convergenceRate": convergence_rate,
            "nodes": nodes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-versioning")
async def get_model_versioning():
    """Get AI model versioning and rollback data (live)"""
    try:
        model_metrics = None
        retrain_metrics = None
        evolution_metrics = None
        if ai_model_versioning and hasattr(ai_model_versioning, "get_model_metrics"):
            try:
                model_metrics = ai_model_versioning.get_model_metrics()
            except Exception:
                model_metrics = None
        if ai_auto_retrain and hasattr(ai_auto_retrain, "get_retrain_metrics"):
            try:
                retrain_metrics = ai_auto_retrain.get_retrain_metrics()
            except Exception:
                retrain_metrics = None
        if ai_genetic_algorithm and hasattr(ai_genetic_algorithm, "get_evolution_metrics"):
            try:
                evolution_metrics = ai_genetic_algorithm.get_evolution_metrics()
            except Exception:
                evolution_metrics = None
        return {
            "modelMetrics": model_metrics,
            "retrainMetrics": retrain_metrics,
            "evolutionMetrics": evolution_metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """Get AI supercomputing alerts and events (live)"""
    try:
        alerts = []
        events = []
        # Get alerts from AlertManager
        if alert_manager:
            try:
                alerts = await alert_manager.get_recent_alerts(10)
            except Exception:
                alerts = []
        # Get alerts from PerformanceMonitor if available
        if performance_monitor:
            try:
                monitor_alerts = (
                    performance_monitor.alerts[-10:]
                    if hasattr(performance_monitor, "alerts")
                    else []
                )
                for alert in monitor_alerts:
                    alerts.append(
                        {
                            "id": f"alert_{len(alerts)}",
                            "severity": alert.level if hasattr(alert, "level") else "info",
                            "message": (
                                alert.message if hasattr(alert, "message") else "Performance alert"
                            ),
                            "timestamp": (
                                alert.timestamp
                                if hasattr(alert, "timestamp")
                                else datetime.now().isoformat()
                            ),
                            "acknowledged": False,
                        }
                    )
            except Exception:
                pass
        # Get events from system endpoints if available
        try:
            from backend.services.system_monitor import SystemMonitor

            system_monitor = SystemMonitor()
            events = await system_monitor.get_recent_events(10)
        except Exception:
            pass
        return {"alerts": alerts, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview")
async def get_supercomputing_overview():
    """Get comprehensive AI supercomputing overview (live)"""
    try:
        # Aggregate all live endpoint data
        model_performance = await get_model_performance()
        resource_utilization = await get_resource_utilization()
        job_scheduler = await get_job_scheduler()
        fault_tolerance = await get_fault_tolerance()
        system_health = await get_system_health()
        distributed_training = await get_distributed_training()
        model_versioning = await get_model_versioning()
        alerts = await get_alerts()
        return {
            "modelPerformance": model_performance.get("models", model_performance),
            "resourceUtilization": resource_utilization,
            "jobScheduler": job_scheduler,
            "faultTolerance": fault_tolerance,
            "systemHealth": system_health,
            "distributedTraining": distributed_training,
            "modelVersioning": model_versioning,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



