#!/usr/bin/env python3
"""
Migration script to replace imperative strategies with functional ones.

This script safely migrates from the imperative strategy system to functional strategies
while maintaining exact API compatibility.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for the migration script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("strategy_migration.log")
        ]
    )


def backup_original_files():
    """Backup original strategy files before migration."""
    backup_dir = Path("backups") / f"strategy_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        "bot/strategy/llm_agent.py",
        "bot/strategy/memory_enhanced_agent.py",
    ]
    
    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            logger.info(f"Backed up {src} to {dst}")
        else:
            logger.warning(f"File not found for backup: {src}")
    
    return backup_dir


def migrate_strategy_files():
    """Replace imperative strategy files with functional versions."""
    
    migrations = [
        {
            "original": "bot/strategy/llm_agent.py",
            "functional": "bot/strategy/llm_agent_functional.py",
            "backup_suffix": ".imperative"
        },
        {
            "original": "bot/strategy/memory_enhanced_agent.py", 
            "functional": "bot/strategy/memory_enhanced_agent_functional.py",
            "backup_suffix": ".imperative"
        }
    ]
    
    for migration in migrations:
        original_path = Path(migration["original"])
        functional_path = Path(migration["functional"])
        backup_path = Path(f"{migration['original']}{migration['backup_suffix']}")
        
        if not original_path.exists():
            logger.warning(f"Original file not found: {original_path}")
            continue
            
        if not functional_path.exists():
            logger.error(f"Functional replacement not found: {functional_path}")
            continue
        
        # Move original to backup location
        shutil.move(original_path, backup_path)
        logger.info(f"Moved {original_path} to {backup_path}")
        
        # Copy functional replacement to original location
        shutil.copy2(functional_path, original_path)
        logger.info(f"Replaced {original_path} with functional implementation")


def verify_migration():
    """Verify that the migration was successful."""
    try:
        # Try to import the migrated modules
        from bot.strategy.llm_agent import LLMAgent
        from bot.strategy.memory_enhanced_agent import MemoryEnhancedLLMAgent
        
        # Check that they are the functional adapters
        from bot.fp.adapters.strategy_adapter import LLMAgentAdapter, MemoryEnhancedLLMAgentAdapter
        
        if LLMAgent is not LLMAgentAdapter:
            logger.error("LLMAgent migration verification failed - not using functional adapter")
            return False
            
        if MemoryEnhancedLLMAgent is not MemoryEnhancedLLMAgentAdapter:
            logger.error("MemoryEnhancedLLMAgent migration verification failed - not using functional adapter")
            return False
        
        logger.info("✅ Migration verification successful - functional strategies are active")
        return True
        
    except ImportError as e:
        logger.error(f"Migration verification failed - import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Migration verification failed - unexpected error: {e}")
        return False


def rollback_migration():
    """Rollback the migration by restoring original files."""
    rollback_files = [
        ("bot/strategy/llm_agent.py.imperative", "bot/strategy/llm_agent.py"),
        ("bot/strategy/memory_enhanced_agent.py.imperative", "bot/strategy/memory_enhanced_agent.py"),
    ]
    
    for backup_path, original_path in rollback_files:
        backup = Path(backup_path)
        original = Path(original_path)
        
        if backup.exists():
            if original.exists():
                original.unlink()  # Remove functional version
            shutil.move(backup, original)
            logger.info(f"Restored {original} from {backup}")
        else:
            logger.warning(f"Backup file not found: {backup}")
    
    logger.info("🔄 Migration rollback completed")


def main():
    """Main migration function."""
    setup_logging()
    
    logger.info("🚀 Starting strategy migration to functional implementation")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        logger.info("🔄 Rolling back migration")
        rollback_migration()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        logger.info("🔍 Verifying migration")
        success = verify_migration()
        sys.exit(0 if success else 1)
    
    try:
        # Step 1: Backup original files
        logger.info("📦 Creating backup of original strategy files")
        backup_dir = backup_original_files()
        
        # Step 2: Perform migration
        logger.info("🔄 Migrating strategy files to functional implementation")
        migrate_strategy_files()
        
        # Step 3: Verify migration
        logger.info("🔍 Verifying migration")
        if verify_migration():
            logger.info("✅ Strategy migration completed successfully!")
            logger.info(f"📦 Original files backed up to: {backup_dir}")
            logger.info("🎯 The system now uses functional strategies with exact API compatibility")
            logger.info("\nTo rollback: python migrate_to_functional_strategies.py --rollback")
        else:
            logger.error("❌ Migration verification failed!")
            logger.info("🔄 Rolling back migration")
            rollback_migration()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.info("🔄 Attempting rollback")
        try:
            rollback_migration()
        except Exception as rollback_error:
            logger.error(f"❌ Rollback also failed: {rollback_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()