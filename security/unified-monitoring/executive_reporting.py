"""
Executive Security Reporting for OPTIMIZE Platform

This module provides comprehensive security reporting and analytics for executive
stakeholders, including security posture assessment, risk analysis, compliance
reporting, and business impact analysis.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import redis
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .alert_orchestrator import Alert

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()


class ExecutiveReport(Base):
    """Database model for executive reports."""

    __tablename__ = "executive_reports"

    id = Column(String, primary_key=True)
    report_type = Column(String, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    generated_at = Column(DateTime, nullable=False)
    generated_by = Column(String)
    report_data = Column(Text)
    file_path = Column(String)
    status = Column(String, default="generated")


class SecurityMetricSnapshot(Base):
    """Database model for security metric snapshots."""

    __tablename__ = "security_metric_snapshots"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    security_score = Column(Float)
    risk_level = Column(String)
    compliance_score = Column(Float)
    threat_count = Column(Integer, default=0)
    vulnerability_count = Column(Integer, default=0)
    incident_count = Column(Integer, default=0)
    uptime_percent = Column(Float, default=100.0)
    performance_impact = Column(Float, default=0.0)
    metadata = Column(Text)


# Enums and Data Classes
class ReportType(Enum):
    """Types of executive reports."""

    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SECURITY = "weekly_security"
    MONTHLY_EXECUTIVE = "monthly_executive"
    QUARTERLY_BOARD = "quarterly_board"
    INCIDENT_ANALYSIS = "incident_analysis"
    COMPLIANCE_AUDIT = "compliance_audit"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_IMPACT = "performance_impact"


class ReportPeriod(Enum):
    """Report time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class SecurityKPI:
    """Key Performance Indicator for security."""

    name: str
    value: float
    unit: str
    trend: str  # "up", "down", "stable"
    trend_percentage: float
    target: float | None = None
    threshold_warning: float | None = None
    threshold_critical: float | None = None
    description: str = ""

    def get_status(self) -> str:
        """Get KPI status based on thresholds."""
        if self.threshold_critical and self.value >= self.threshold_critical:
            return "critical"
        if self.threshold_warning and self.value >= self.threshold_warning:
            return "warning"
        if self.target and abs(self.value - self.target) / self.target < 0.1:
            return "on_target"
        return "normal"


@dataclass
class ThreatLandscape:
    """Threat landscape analysis."""

    total_threats: int
    threats_by_category: dict[str, int]
    threats_by_severity: dict[str, int]
    top_attack_vectors: list[tuple[str, int]]
    geographic_distribution: dict[str, int]
    trend_analysis: dict[str, float]
    threat_intelligence_summary: str


@dataclass
class ComplianceStatus:
    """Compliance framework status."""

    framework: str
    overall_score: float
    control_scores: dict[str, float]
    passed_controls: int
    failed_controls: int
    remediation_items: list[str]
    certification_status: str
    next_audit_date: datetime | None = None


@dataclass
class BusinessImpactAnalysis:
    """Business impact analysis."""

    trading_uptime: float
    performance_impact: float
    security_incidents: int
    estimated_prevented_losses: float
    security_investment_roi: float
    compliance_cost_avoidance: float
    reputation_risk_score: float
    operational_efficiency: float


@dataclass
class ExecutiveReportData:
    """Complete executive report data."""

    report_id: str = field(default_factory=lambda: str(uuid4()))
    report_type: ReportType = ReportType.MONTHLY_EXECUTIVE
    period_start: datetime = field(
        default_factory=lambda: datetime.utcnow() - timedelta(days=30)
    )
    period_end: datetime = field(default_factory=datetime.utcnow)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Executive Summary
    executive_summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Security KPIs
    security_kpis: list[SecurityKPI] = field(default_factory=list)
    security_score: float = 0.0
    risk_level: str = "unknown"

    # Threat Analysis
    threat_landscape: ThreatLandscape | None = None

    # Compliance
    compliance_status: list[ComplianceStatus] = field(default_factory=list)
    overall_compliance_score: float = 0.0

    # Business Impact
    business_impact: BusinessImpactAnalysis | None = None

    # Performance
    system_performance: dict[str, Any] = field(default_factory=dict)

    # Trends and Forecasting
    trend_analysis: dict[str, Any] = field(default_factory=dict)
    risk_forecast: dict[str, Any] = field(default_factory=dict)


class SecurityMetricsCollector:
    """Collects and aggregates security metrics for reporting."""

    def __init__(self, db_session):
        self.session = db_session

    async def collect_security_kpis(
        self, period_start: datetime, period_end: datetime
    ) -> list[SecurityKPI]:
        """Collect key security performance indicators."""
        kpis = []

        try:
            # Security Score KPI
            security_score = await self._calculate_security_score(
                period_start, period_end
            )
            security_score_kpi = SecurityKPI(
                name="Security Score",
                value=security_score,
                unit="percent",
                trend=await self._calculate_trend(
                    security_score, "security_score", period_start
                ),
                trend_percentage=await self._calculate_trend_percentage(
                    security_score, "security_score", period_start
                ),
                target=95.0,
                threshold_warning=80.0,
                threshold_critical=70.0,
                description="Overall security posture score based on multiple factors",
            )
            kpis.append(security_score_kpi)

            # Incident Response Time KPI
            response_time = await self._calculate_avg_response_time(
                period_start, period_end
            )
            response_time_kpi = SecurityKPI(
                name="Incident Response Time",
                value=response_time,
                unit="minutes",
                trend=await self._calculate_trend(
                    response_time, "response_time", period_start
                ),
                trend_percentage=await self._calculate_trend_percentage(
                    response_time, "response_time", period_start
                ),
                target=30.0,
                threshold_warning=60.0,
                threshold_critical=120.0,
                description="Average time to respond to security incidents",
            )
            kpis.append(response_time_kpi)

            # Threat Detection Rate KPI
            detection_rate = await self._calculate_threat_detection_rate(
                period_start, period_end
            )
            detection_rate_kpi = SecurityKPI(
                name="Threat Detection Rate",
                value=detection_rate,
                unit="percent",
                trend=await self._calculate_trend(
                    detection_rate, "detection_rate", period_start
                ),
                trend_percentage=await self._calculate_trend_percentage(
                    detection_rate, "detection_rate", period_start
                ),
                target=98.0,
                threshold_warning=90.0,
                threshold_critical=80.0,
                description="Percentage of threats successfully detected",
            )
            kpis.append(detection_rate_kpi)

            # System Uptime KPI
            uptime = await self._calculate_system_uptime(period_start, period_end)
            uptime_kpi = SecurityKPI(
                name="System Uptime",
                value=uptime,
                unit="percent",
                trend=await self._calculate_trend(uptime, "uptime", period_start),
                trend_percentage=await self._calculate_trend_percentage(
                    uptime, "uptime", period_start
                ),
                target=99.9,
                threshold_warning=99.5,
                threshold_critical=99.0,
                description="Trading system availability percentage",
            )
            kpis.append(uptime_kpi)

            # False Positive Rate KPI
            false_positive_rate = await self._calculate_false_positive_rate(
                period_start, period_end
            )
            fp_rate_kpi = SecurityKPI(
                name="False Positive Rate",
                value=false_positive_rate,
                unit="percent",
                trend=await self._calculate_trend(
                    false_positive_rate, "false_positive_rate", period_start
                ),
                trend_percentage=await self._calculate_trend_percentage(
                    false_positive_rate, "false_positive_rate", period_start
                ),
                target=5.0,
                threshold_warning=15.0,
                threshold_critical=25.0,
                description="Percentage of false positive security alerts",
            )
            kpis.append(fp_rate_kpi)

        except Exception as e:
            logger.error(f"Error collecting security KPIs: {e}")

        return kpis

    async def _calculate_security_score(self, start: datetime, end: datetime) -> float:
        """Calculate overall security score."""
        # TODO: Implement sophisticated security score calculation
        # For now, return a weighted average of various factors

        # Get alert counts by severity
        critical_alerts = (
            self.session.query(Alert)
            .filter(Alert.severity == "critical", Alert.created_at.between(start, end))
            .count()
        )

        high_alerts = (
            self.session.query(Alert)
            .filter(Alert.severity == "high", Alert.created_at.between(start, end))
            .count()
        )

        # Base score starts at 100
        score = 100.0

        # Deduct points for alerts
        score -= critical_alerts * 10
        score -= high_alerts * 5

        # Ensure score is between 0 and 100
        return max(0, min(100, score))

    async def _calculate_avg_response_time(
        self, start: datetime, end: datetime
    ) -> float:
        """Calculate average incident response time."""
        # TODO: Implement actual response time calculation
        return 25.5  # Mock value in minutes

    async def _calculate_threat_detection_rate(
        self, start: datetime, end: datetime
    ) -> float:
        """Calculate threat detection rate."""
        # TODO: Implement actual detection rate calculation
        return 96.8  # Mock value

    async def _calculate_system_uptime(self, start: datetime, end: datetime) -> float:
        """Calculate system uptime percentage."""
        # TODO: Implement actual uptime calculation
        return 99.95  # Mock value

    async def _calculate_false_positive_rate(
        self, start: datetime, end: datetime
    ) -> float:
        """Calculate false positive rate."""
        # TODO: Implement actual false positive rate calculation
        return 8.2  # Mock value

    async def _calculate_trend(
        self, current_value: float, metric_name: str, period_start: datetime
    ) -> str:
        """Calculate trend direction for a metric."""
        # TODO: Implement trend calculation by comparing with previous period
        return "up"  # Mock value

    async def _calculate_trend_percentage(
        self, current_value: float, metric_name: str, period_start: datetime
    ) -> float:
        """Calculate trend percentage change."""
        # TODO: Implement actual trend percentage calculation
        return 2.5  # Mock value


class ThreatAnalyzer:
    """Analyzes threat landscape and security events."""

    def __init__(self, db_session):
        self.session = db_session

    async def analyze_threat_landscape(
        self, period_start: datetime, period_end: datetime
    ) -> ThreatLandscape:
        """Analyze threat landscape for the specified period."""
        try:
            # Get all alerts for the period
            alerts = (
                self.session.query(Alert)
                .filter(Alert.created_at.between(period_start, period_end))
                .all()
            )

            # Categorize threats
            threats_by_category = {}
            threats_by_severity = {}

            for alert in alerts:
                # TODO: Extract category from alert data
                category = "unknown"  # Placeholder
                threats_by_category[category] = threats_by_category.get(category, 0) + 1

                severity = alert.severity
                threats_by_severity[severity] = threats_by_severity.get(severity, 0) + 1

            # Top attack vectors (mock data)
            top_attack_vectors = [
                ("Container Escape", 15),
                ("Privilege Escalation", 12),
                ("Network Intrusion", 8),
                ("Data Exfiltration", 5),
                ("Credential Theft", 3),
            ]

            # Geographic distribution (mock data)
            geographic_distribution = {
                "Unknown": 25,
                "US": 10,
                "China": 8,
                "Russia": 5,
                "Other": 2,
            }

            # Trend analysis (mock data)
            trend_analysis = {
                "total_threats_change": -15.2,
                "severity_trend": "improving",
                "new_attack_vectors": 2,
            }

            threat_landscape = ThreatLandscape(
                total_threats=len(alerts),
                threats_by_category=threats_by_category,
                threats_by_severity=threats_by_severity,
                top_attack_vectors=top_attack_vectors,
                geographic_distribution=geographic_distribution,
                trend_analysis=trend_analysis,
                threat_intelligence_summary="Threat activity has decreased by 15% compared to the previous period. Container-based attacks remain the primary concern.",
            )

            return threat_landscape

        except Exception as e:
            logger.error(f"Error analyzing threat landscape: {e}")
            return ThreatLandscape(
                total_threats=0,
                threats_by_category={},
                threats_by_severity={},
                top_attack_vectors=[],
                geographic_distribution={},
                trend_analysis={},
                threat_intelligence_summary="Error analyzing threat data",
            )


class ComplianceAnalyzer:
    """Analyzes compliance status across frameworks."""

    def __init__(self, db_session):
        self.session = db_session

    async def analyze_compliance_status(self) -> list[ComplianceStatus]:
        """Analyze compliance status for all frameworks."""
        compliance_statuses = []

        try:
            # SOC 2 Type II Compliance
            soc2_status = ComplianceStatus(
                framework="SOC 2 Type II",
                overall_score=98.5,
                control_scores={
                    "Access Control": 99.2,
                    "Data Protection": 98.1,
                    "Monitoring": 99.0,
                    "Incident Response": 97.8,
                    "Change Management": 98.5,
                },
                passed_controls=47,
                failed_controls=1,
                remediation_items=[
                    "Update password complexity requirements",
                    "Enhance audit log retention policies",
                ],
                certification_status="Compliant",
                next_audit_date=datetime.utcnow() + timedelta(days=90),
            )
            compliance_statuses.append(soc2_status)

            # PCI DSS Compliance
            pci_status = ComplianceStatus(
                framework="PCI DSS",
                overall_score=96.8,
                control_scores={
                    "Network Security": 98.0,
                    "Access Control": 95.5,
                    "Data Protection": 97.2,
                    "Vulnerability Management": 96.0,
                    "Monitoring": 98.5,
                },
                passed_controls=42,
                failed_controls=2,
                remediation_items=[
                    "Implement additional network segmentation",
                    "Enhance cardholder data encryption",
                ],
                certification_status="Compliant",
                next_audit_date=datetime.utcnow() + timedelta(days=180),
            )
            compliance_statuses.append(pci_status)

            # GDPR Compliance
            gdpr_status = ComplianceStatus(
                framework="GDPR",
                overall_score=94.2,
                control_scores={
                    "Data Processing": 95.0,
                    "Consent Management": 93.5,
                    "Data Subject Rights": 94.8,
                    "Data Protection": 95.2,
                    "Breach Notification": 92.0,
                },
                passed_controls=38,
                failed_controls=3,
                remediation_items=[
                    "Enhance data subject access request procedures",
                    "Update privacy notice documentation",
                    "Implement automated data deletion workflows",
                ],
                certification_status="Compliant",
                next_audit_date=datetime.utcnow() + timedelta(days=365),
            )
            compliance_statuses.append(gdpr_status)

        except Exception as e:
            logger.error(f"Error analyzing compliance status: {e}")

        return compliance_statuses


class BusinessImpactCalculator:
    """Calculates business impact of security measures."""

    def __init__(self, db_session):
        self.session = db_session

    async def calculate_business_impact(
        self, period_start: datetime, period_end: datetime
    ) -> BusinessImpactAnalysis:
        """Calculate business impact analysis."""
        try:
            # Calculate trading uptime
            trading_uptime = await self._calculate_trading_uptime(
                period_start, period_end
            )

            # Calculate performance impact
            performance_impact = await self._calculate_performance_impact(
                period_start, period_end
            )

            # Count security incidents
            incident_count = (
                self.session.query(Alert)
                .filter(
                    Alert.created_at.between(period_start, period_end),
                    Alert.severity.in_(["critical", "high"]),
                )
                .count()
            )

            # Estimate prevented losses (based on blocked threats)
            prevented_losses = await self._estimate_prevented_losses(
                period_start, period_end
            )

            # Calculate security ROI
            security_roi = await self._calculate_security_roi(period_start, period_end)

            # Calculate compliance cost avoidance
            compliance_savings = await self._calculate_compliance_cost_avoidance()

            # Assess reputation risk
            reputation_risk = await self._assess_reputation_risk(
                period_start, period_end
            )

            # Calculate operational efficiency
            operational_efficiency = await self._calculate_operational_efficiency(
                period_start, period_end
            )

            return BusinessImpactAnalysis(
                trading_uptime=trading_uptime,
                performance_impact=performance_impact,
                security_incidents=incident_count,
                estimated_prevented_losses=prevented_losses,
                security_investment_roi=security_roi,
                compliance_cost_avoidance=compliance_savings,
                reputation_risk_score=reputation_risk,
                operational_efficiency=operational_efficiency,
            )

        except Exception as e:
            logger.error(f"Error calculating business impact: {e}")
            return BusinessImpactAnalysis(
                trading_uptime=0.0,
                performance_impact=0.0,
                security_incidents=0,
                estimated_prevented_losses=0.0,
                security_investment_roi=0.0,
                compliance_cost_avoidance=0.0,
                reputation_risk_score=0.0,
                operational_efficiency=0.0,
            )

    async def _calculate_trading_uptime(self, start: datetime, end: datetime) -> float:
        """Calculate trading system uptime."""
        # TODO: Implement actual uptime calculation
        return 99.95

    async def _calculate_performance_impact(
        self, start: datetime, end: datetime
    ) -> float:
        """Calculate performance impact of security tools."""
        # TODO: Query performance metrics
        return 1.8  # Mock value in percent

    async def _estimate_prevented_losses(self, start: datetime, end: datetime) -> float:
        """Estimate financial losses prevented by security measures."""
        # TODO: Implement sophisticated loss prevention calculation
        return 245000.0  # Mock value in USD

    async def _calculate_security_roi(self, start: datetime, end: datetime) -> float:
        """Calculate return on investment for security measures."""
        # TODO: Calculate actual ROI based on costs and benefits
        return 2150.0  # Mock value in percent

    async def _calculate_compliance_cost_avoidance(self) -> float:
        """Calculate cost avoidance from compliance adherence."""
        # TODO: Calculate potential fine avoidance
        return 150000.0  # Mock value in USD

    async def _assess_reputation_risk(self, start: datetime, end: datetime) -> float:
        """Assess reputation risk score."""
        # TODO: Implement reputation risk assessment
        return 15.2  # Mock value (lower is better)

    async def _calculate_operational_efficiency(
        self, start: datetime, end: datetime
    ) -> float:
        """Calculate operational efficiency score."""
        # TODO: Calculate efficiency based on automation and response times
        return 92.5  # Mock value in percent


class ReportGenerator:
    """Generates various types of executive reports."""

    def __init__(self, template_dir: str = "templates"):
        self.template_env = Environment(loader=FileSystemLoader(template_dir))

    async def generate_monthly_executive_report(
        self, report_data: ExecutiveReportData
    ) -> str:
        """Generate monthly executive report in HTML format."""
        try:
            template = self.template_env.get_template("monthly_executive_report.html")

            # Prepare template context
            context = {
                "report": report_data,
                "generated_date": report_data.generated_at.strftime("%B %d, %Y"),
                "period_description": f"{report_data.period_start.strftime('%B %d')} - {report_data.period_end.strftime('%B %d, %Y')}",
                "charts": await self._generate_charts(report_data),
            }

            # Render the report
            html_content = template.render(context)

            return html_content

        except Exception as e:
            logger.error(f"Error generating monthly executive report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    async def generate_daily_summary_report(
        self, report_data: ExecutiveReportData
    ) -> str:
        """Generate daily summary report."""
        try:
            # Simple text-based daily summary
            summary = f"""
DAILY SECURITY SUMMARY - {report_data.period_end.strftime("%Y-%m-%d")}

SECURITY POSTURE:
- Security Score: {report_data.security_score:.1f}%
- Risk Level: {report_data.risk_level.upper()}
- Active Incidents: {len([kpi for kpi in report_data.security_kpis if "incident" in kpi.name.lower()])}

KEY METRICS:
"""

            for kpi in report_data.security_kpis[:5]:  # Top 5 KPIs
                status_emoji = {
                    "critical": "🔴",
                    "warning": "🟡",
                    "normal": "🟢",
                    "on_target": "✅",
                }.get(kpi.get_status(), "⚪")
                summary += f"- {kpi.name}: {kpi.value}{kpi.unit} {status_emoji}\n"

            if report_data.key_findings:
                summary += "\nKEY FINDINGS:\n"
                for finding in report_data.key_findings:
                    summary += f"- {finding}\n"

            if report_data.recommendations:
                summary += "\nRECOMMENDATIONS:\n"
                for rec in report_data.recommendations:
                    summary += f"- {rec}\n"

            summary += f"\nGenerated: {report_data.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"

            return summary

        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            return f"Error generating daily summary: {e}"

    async def _generate_charts(
        self, report_data: ExecutiveReportData
    ) -> dict[str, str]:
        """Generate charts for the report."""
        charts = {}

        try:
            # Security KPIs chart
            if report_data.security_kpis:
                kpi_chart = await self._create_kpi_chart(report_data.security_kpis)
                charts["kpi_chart"] = kpi_chart

            # Threat landscape chart
            if report_data.threat_landscape:
                threat_chart = await self._create_threat_chart(
                    report_data.threat_landscape
                )
                charts["threat_chart"] = threat_chart

            # Compliance chart
            if report_data.compliance_status:
                compliance_chart = await self._create_compliance_chart(
                    report_data.compliance_status
                )
                charts["compliance_chart"] = compliance_chart

        except Exception as e:
            logger.error(f"Error generating charts: {e}")

        return charts

    async def _create_kpi_chart(self, kpis: list[SecurityKPI]) -> str:
        """Create KPI dashboard chart."""
        # TODO: Create actual charts using matplotlib/plotly
        return "kpi_chart_placeholder.png"

    async def _create_threat_chart(self, threat_landscape: ThreatLandscape) -> str:
        """Create threat landscape visualization."""
        # TODO: Create threat visualization
        return "threat_chart_placeholder.png"

    async def _create_compliance_chart(
        self, compliance_status: list[ComplianceStatus]
    ) -> str:
        """Create compliance status chart."""
        # TODO: Create compliance visualization
        return "compliance_chart_placeholder.png"


class ExecutiveReportingSystem:
    """Main executive reporting system."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_url: str = "sqlite:///executive_reports.db",
    ):
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Components
        self.report_generator = ReportGenerator()

        # Internal state
        self.running = False
        self.scheduled_reports = {}

        # Report schedules
        self.report_schedules = {
            ReportType.DAILY_SUMMARY: timedelta(days=1),
            ReportType.WEEKLY_SECURITY: timedelta(weeks=1),
            ReportType.MONTHLY_EXECUTIVE: timedelta(days=30),
            ReportType.QUARTERLY_BOARD: timedelta(days=90),
        }

    async def start(self):
        """Start the executive reporting system."""
        self.running = True
        logger.info("Starting Executive Reporting System")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._schedule_reports()),
            asyncio.create_task(self._generate_scheduled_reports()),
            asyncio.create_task(self._cleanup_old_reports()),
        ]

        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop the executive reporting system."""
        self.running = False
        logger.info("Executive Reporting System stopped")

    async def generate_report(
        self,
        report_type: ReportType,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> ExecutiveReportData:
        """Generate an executive report."""
        try:
            # Set default period if not provided
            if period_end is None:
                period_end = datetime.utcnow()

            if period_start is None:
                if report_type == ReportType.DAILY_SUMMARY:
                    period_start = period_end - timedelta(days=1)
                elif report_type == ReportType.WEEKLY_SECURITY:
                    period_start = period_end - timedelta(weeks=1)
                elif report_type == ReportType.MONTHLY_EXECUTIVE:
                    period_start = period_end - timedelta(days=30)
                elif report_type == ReportType.QUARTERLY_BOARD:
                    period_start = period_end - timedelta(days=90)
                else:
                    period_start = period_end - timedelta(days=7)

            # Initialize report data
            report_data = ExecutiveReportData(
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
            )

            # Collect data from various sources
            session = self.Session()
            try:
                # Collect security metrics
                metrics_collector = SecurityMetricsCollector(session)
                report_data.security_kpis = (
                    await metrics_collector.collect_security_kpis(
                        period_start, period_end
                    )
                )

                # Calculate security score
                if report_data.security_kpis:
                    security_score_kpi = next(
                        (
                            kpi
                            for kpi in report_data.security_kpis
                            if kpi.name == "Security Score"
                        ),
                        None,
                    )
                    if security_score_kpi:
                        report_data.security_score = security_score_kpi.value

                # Determine risk level
                report_data.risk_level = self._determine_risk_level(
                    report_data.security_score
                )

                # Analyze threat landscape
                threat_analyzer = ThreatAnalyzer(session)
                report_data.threat_landscape = (
                    await threat_analyzer.analyze_threat_landscape(
                        period_start, period_end
                    )
                )

                # Analyze compliance
                compliance_analyzer = ComplianceAnalyzer(session)
                report_data.compliance_status = (
                    await compliance_analyzer.analyze_compliance_status()
                )

                # Calculate overall compliance score
                if report_data.compliance_status:
                    report_data.overall_compliance_score = sum(
                        cs.overall_score for cs in report_data.compliance_status
                    ) / len(report_data.compliance_status)

                # Calculate business impact
                impact_calculator = BusinessImpactCalculator(session)
                report_data.business_impact = (
                    await impact_calculator.calculate_business_impact(
                        period_start, period_end
                    )
                )

                # Generate executive summary and recommendations
                report_data.executive_summary = await self._generate_executive_summary(
                    report_data
                )
                report_data.key_findings = await self._extract_key_findings(report_data)
                report_data.recommendations = await self._generate_recommendations(
                    report_data
                )

            finally:
                session.close()

            # Store report in database
            await self._store_report(report_data)

            logger.info(
                f"Generated {report_type.value} report for period {period_start} to {period_end}"
            )
            return report_data

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def _determine_risk_level(self, security_score: float) -> str:
        """Determine risk level based on security score."""
        if security_score >= 90:
            return "low"
        if security_score >= 75:
            return "medium"
        if security_score >= 60:
            return "high"
        return "critical"

    async def _generate_executive_summary(
        self, report_data: ExecutiveReportData
    ) -> str:
        """Generate executive summary text."""
        try:
            summary_parts = []

            # Security posture summary
            summary_parts.append(
                f"The organization maintains a {report_data.risk_level} risk posture with a security score of {report_data.security_score:.1f}%."
            )

            # Threat summary
            if report_data.threat_landscape:
                summary_parts.append(
                    f"During this period, {report_data.threat_landscape.total_threats} security events were detected and analyzed."
                )

            # Business impact summary
            if report_data.business_impact:
                summary_parts.append(
                    f"Trading system uptime was maintained at {report_data.business_impact.trading_uptime:.2f}% with minimal performance impact of {report_data.business_impact.performance_impact:.1f}%."
                )

            # Compliance summary
            if report_data.compliance_status:
                compliant_frameworks = sum(
                    1
                    for cs in report_data.compliance_status
                    if cs.certification_status == "Compliant"
                )
                summary_parts.append(
                    f"The organization remains compliant with {compliant_frameworks} out of {len(report_data.compliance_status)} regulatory frameworks."
                )

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Executive summary generation failed."

    async def _extract_key_findings(
        self, report_data: ExecutiveReportData
    ) -> list[str]:
        """Extract key findings from report data."""
        findings = []

        try:
            # Analyze KPI trends
            for kpi in report_data.security_kpis:
                if kpi.get_status() == "critical":
                    findings.append(
                        f"{kpi.name} is at critical level ({kpi.value}{kpi.unit})"
                    )
                elif kpi.trend == "down" and kpi.trend_percentage < -10:
                    findings.append(
                        f"{kpi.name} has declined significantly by {abs(kpi.trend_percentage):.1f}%"
                    )
                elif kpi.trend == "up" and kpi.trend_percentage > 10:
                    findings.append(
                        f"{kpi.name} has improved by {kpi.trend_percentage:.1f}%"
                    )

            # Threat landscape findings
            if report_data.threat_landscape:
                if report_data.threat_landscape.total_threats > 100:
                    findings.append(
                        f"High threat activity detected with {report_data.threat_landscape.total_threats} events"
                    )

                if report_data.threat_landscape.top_attack_vectors:
                    top_attack = report_data.threat_landscape.top_attack_vectors[0]
                    findings.append(
                        f"Primary attack vector is {top_attack[0]} with {top_attack[1]} incidents"
                    )

            # Compliance findings
            for compliance in report_data.compliance_status:
                if compliance.failed_controls > 0:
                    findings.append(
                        f"{compliance.framework} has {compliance.failed_controls} failed controls requiring attention"
                    )

        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")

        return findings

    async def _generate_recommendations(
        self, report_data: ExecutiveReportData
    ) -> list[str]:
        """Generate recommendations based on report data."""
        recommendations = []

        try:
            # Security score-based recommendations
            if report_data.security_score < 80:
                recommendations.append(
                    "Consider increasing security monitoring coverage and response capabilities"
                )

            # Performance-based recommendations
            if (
                report_data.business_impact
                and report_data.business_impact.performance_impact > 5
            ):
                recommendations.append(
                    "Optimize security tool configurations to reduce performance impact on trading operations"
                )

            # Compliance-based recommendations
            for compliance in report_data.compliance_status:
                if compliance.failed_controls > 0:
                    recommendations.append(
                        f"Address {compliance.framework} compliance gaps: {', '.join(compliance.remediation_items[:2])}"
                    )

            # Threat-based recommendations
            if (
                report_data.threat_landscape
                and report_data.threat_landscape.total_threats > 50
            ):
                recommendations.append(
                    "Enhance threat detection rules to reduce false positives and improve signal-to-noise ratio"
                )

            # Generic recommendations
            if len(recommendations) == 0:
                recommendations.append(
                    "Continue current security practices and monitor for emerging threats"
                )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        return recommendations

    async def _store_report(self, report_data: ExecutiveReportData):
        """Store report data in database."""
        session = self.Session()
        try:
            report = ExecutiveReport(
                id=report_data.report_id,
                report_type=report_data.report_type.value,
                period_start=report_data.period_start,
                period_end=report_data.period_end,
                generated_at=report_data.generated_at,
                generated_by="system",
                report_data=json.dumps(
                    {
                        "security_score": report_data.security_score,
                        "risk_level": report_data.risk_level,
                        "executive_summary": report_data.executive_summary,
                        "key_findings": report_data.key_findings,
                        "recommendations": report_data.recommendations,
                        "kpi_count": len(report_data.security_kpis),
                        "threat_count": (
                            report_data.threat_landscape.total_threats
                            if report_data.threat_landscape
                            else 0
                        ),
                        "compliance_score": report_data.overall_compliance_score,
                    }
                ),
            )

            session.add(report)
            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing report: {e}")
        finally:
            session.close()

    async def _schedule_reports(self):
        """Schedule automatic report generation."""
        while self.running:
            try:
                current_time = datetime.utcnow()

                # Check if any reports need to be generated
                for report_type, interval in self.report_schedules.items():
                    last_generated = self.scheduled_reports.get(report_type)

                    if (
                        last_generated is None
                        or current_time - last_generated >= interval
                    ):
                        logger.info(f"Scheduling {report_type.value} report generation")

                        # Generate report
                        try:
                            await self.generate_report(report_type)
                            self.scheduled_reports[report_type] = current_time
                        except Exception as e:
                            logger.error(
                                f"Error generating scheduled report {report_type.value}: {e}"
                            )

                # Check every hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in report scheduling: {e}")
                await asyncio.sleep(3600)

    async def _generate_scheduled_reports(self):
        """Process report generation queue."""
        while self.running:
            try:
                # TODO: Implement report generation queue processing
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in scheduled report generation: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_reports(self):
        """Clean up old reports."""
        while self.running:
            try:
                # Clean up reports older than 1 year
                session = self.Session()
                cutoff_date = datetime.utcnow() - timedelta(days=365)

                deleted_count = (
                    session.query(ExecutiveReport)
                    .filter(ExecutiveReport.generated_at < cutoff_date)
                    .delete()
                )

                session.commit()
                session.close()

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old reports")

                # Sleep for 24 hours
                await asyncio.sleep(86400)

            except Exception as e:
                logger.error(f"Error cleaning up old reports: {e}")
                await asyncio.sleep(3600)

    async def get_reporting_status(self) -> dict[str, Any]:
        """Get current reporting system status."""
        try:
            session = self.Session()

            # Get report counts by type
            report_counts = {}
            for report_type in ReportType:
                count = (
                    session.query(ExecutiveReport)
                    .filter(
                        ExecutiveReport.report_type == report_type.value,
                        ExecutiveReport.generated_at
                        >= datetime.utcnow() - timedelta(days=30),
                    )
                    .count()
                )
                report_counts[report_type.value] = count

            session.close()

            status = {
                "running": self.running,
                "scheduled_reports": {
                    rt.value: self.scheduled_reports.get(rt) for rt in ReportType
                },
                "report_counts_30d": report_counts,
                "last_update": datetime.utcnow().isoformat(),
            }

            return status

        except Exception as e:
            logger.error(f"Error getting reporting status: {e}")
            return {"error": str(e)}


# Factory function
def create_executive_reporting_system(
    redis_url: str = "redis://localhost:6379",
    db_url: str = "sqlite:///executive_reports.db",
) -> ExecutiveReportingSystem:
    """Create an executive reporting system instance."""
    return ExecutiveReportingSystem(redis_url, db_url)


if __name__ == "__main__":

    async def main():
        reporting_system = create_executive_reporting_system()

        # Generate a sample monthly report
        report = await reporting_system.generate_report(ReportType.MONTHLY_EXECUTIVE)
        print(f"Generated report: {report.report_id}")
        print(f"Security Score: {report.security_score}")
        print(f"Risk Level: {report.risk_level}")
        print(f"Executive Summary: {report.executive_summary}")

    asyncio.run(main())
