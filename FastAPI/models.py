from sqlalchemy import Column, Integer, String, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PlayerPrecompute(Base):
    __tablename__ = "player_precomputes"

    Team = Column("Team", String, nullable=True)
    Player = Column("Player", String, primary_key=True, nullable=False) # Use as primary key unless you have a better ID

    Days_since_last_game = Column("Days_since_last_game", Integer, nullable=True)
    Missed_Game_Time = Column("Missed_Game_Time", Boolean, nullable=True)
    ToG_Last = Column("ToG_Last", Float, nullable=True)
    ToG_Season_Avg = Column("ToG_Season_Avg", Float, nullable=True)
    Games_This_Season = Column("Games_This_Season", Integer, nullable=True)
    Next_Opponent = Column("Next_Opponent", String, nullable=True)
    Next_Venue_Home = Column("Next_Venue_Home", Boolean, nullable=True)

    Prob_20_Disposals = Column("Prob_20_Disposals", Float, nullable=True)
    Prob_25_Disposals = Column("Prob_25_Disposals", Float, nullable=True)
    Prob_30_Disposals = Column("Prob_30_Disposals", Float, nullable=True)

    Disposal_Season_Avg = Column("Disposal_Season_Avg", Float, nullable=True)
    Disposal_Season_Median = Column("Disposal_Season_Median", Float, nullable=True)
    Disposal_Season_High = Column("Disposal_Season_High", Integer, nullable=True)
    Disposal_Season_Low = Column("Disposal_Season_Low", Integer, nullable=True)
    Disposal_Season_Variance = Column("Disposal_Season_Variance", Float, nullable=True)
    Disposal_Season_DropConsistency = Column("Disposal_Season_DropConsistency", Float, nullable=True)

    Disposal_Season_Dry_Avg = Column("Disposal_Season_Dry_Avg", Float, nullable=True)
    Disposal_Season_Dry_Median = Column("Disposal_Season_Dry_Median", Float, nullable=True)
    Disposal_Season_Dry_High = Column("Disposal_Season_Dry_High", Integer, nullable=True)
    Disposal_Season_Dry_Low = Column("Disposal_Season_Dry_Low", Integer, nullable=True)
    Disposal_Season_Dry_Variance = Column("Disposal_Season_Dry_Variance", Float, nullable=True)

    Disposal_Season_Wet_Avg = Column("Disposal_Season_Wet_Avg", Float, nullable=True)
    Disposal_Season_Wet_Median = Column("Disposal_Season_Wet_Median", Float, nullable=True)
    Disposal_Season_Wet_High = Column("Disposal_Season_Wet_High", Integer, nullable=True)
    Disposal_Season_Wet_Low = Column("Disposal_Season_Wet_Low", Integer, nullable=True)
    Disposal_Season_Wet_Variance = Column("Disposal_Season_Wet_Variance", Float, nullable=True)
    
    Disposal_Season_Day_Avg = Column("Disposal_Season_Day_Avg", Float, nullable=True)
    Disposal_Season_Day_Median = Column("Disposal_Season_Day_Median", Float, nullable=True)
    Disposal_Season_Day_High = Column("Disposal_Season_Day_High", Integer, nullable=True)
    Disposal_Season_Day_Low = Column("Disposal_Season_Day_Low", Integer, nullable=True)
    Disposal_Season_Day_Variance = Column("Disposal_Season_Day_Variance", Float, nullable=True)
    
    Disposal_Season_Twilight_Avg = Column("Disposal_Season_Twilight_Avg", Float, nullable=True)
    Disposal_Season_Twilight_Median = Column("Disposal_Season_Twilight_Median", Float, nullable=True)
    Disposal_Season_Twilight_High = Column("Disposal_Season_Twilight_High", Integer, nullable=True)
    Disposal_Season_Twilight_Low = Column("Disposal_Season_Twilight_Low", Integer, nullable=True)
    Disposal_Season_Twilight_Variance = Column("Disposal_Season_Twilight_Variance", Float, nullable=True)
    
    Disposal_Season_Night_Avg = Column("Disposal_Season_Night_Avg", Float, nullable=True)
    Disposal_Season_Night_Median = Column("Disposal_Season_Night_Median", Float, nullable=True)
    Disposal_Season_Night_High = Column("Disposal_Season_Night_High", Integer, nullable=True)
    Disposal_Season_Night_Low = Column("Disposal_Season_Night_Low", Integer, nullable=True)
    Disposal_Season_Night_Variance = Column("Disposal_Season_Night_Variance", Float, nullable=True)
    
    Disposal_Season_Home_Avg = Column("Disposal_Season_Home_Avg", Float, nullable=True)
    Disposal_Season_Home_Median = Column("Disposal_Season_Home_Median", Float, nullable=True)
    Disposal_Season_Home_High = Column("Disposal_Season_Home_High", Integer, nullable=True)
    Disposal_Season_Home_Low = Column("Disposal_Season_Home_Low", Integer, nullable=True)
    Disposal_Season_Home_Variance = Column("Disposal_Season_Home_Variance", Float, nullable=True)
    
    Disposal_Season_Away_Avg = Column("Disposal_Season_Away_Avg", Float, nullable=True)
    Disposal_Season_Away_Median = Column("Disposal_Season_Away_Median", Float, nullable=True)
    Disposal_Season_Away_High = Column("Disposal_Season_Away_High", Integer, nullable=True)
    Disposal_Season_Away_Low = Column("Disposal_Season_Away_Low", Integer, nullable=True)
    Disposal_Season_Away_Variance = Column("Disposal_Season_Away_Variance", Float, nullable=True)
    
    Disposal_3_Avg = Column("Disposal_3_Avg", Float, nullable=True)
    Disposal_3_Median = Column("Disposal_3_Median", Float, nullable=True)
    Disposal_3_High = Column("Disposal_3_High", Integer, nullable=True)
    Disposal_3_Low = Column("Disposal_3_Low", Integer, nullable=True)
    Disposal_3_Variance = Column("Disposal_3_Variance", Float, nullable=True)
    
    Disposal_3_Dry_Avg = Column("Disposal_3_Dry_Avg", Float, nullable=True)
    Disposal_3_Dry_Median = Column("Disposal_3_Dry_Median", Float, nullable=True)
    Disposal_3_Dry_High = Column("Disposal_3_Dry_High", Integer, nullable=True)
    Disposal_3_Dry_Low = Column("Disposal_3_Dry_Low", Integer, nullable=True)
    Disposal_3_Dry_Variance = Column("Disposal_3_Dry_Variance", Float, nullable=True)
    
    Disposal_3_Wet_Avg = Column("Disposal_3_Wet_Avg", Float, nullable=True)
    Disposal_3_Wet_Median = Column("Disposal_3_Wet_Median", Float, nullable=True)
    Disposal_3_Wet_High = Column("Disposal_3_Wet_High", Integer, nullable=True)
    Disposal_3_Wet_Low = Column("Disposal_3_Wet_Low", Integer, nullable=True)
    Disposal_3_Wet_Variance = Column("Disposal_3_Wet_Variance", Float, nullable=True)
    
    Disposal_3_Day_Avg = Column("Disposal_3_Day_Avg", Float, nullable=True)
    Disposal_3_Day_Median = Column("Disposal_3_Day_Median", Float, nullable=True)
    Disposal_3_Day_High = Column("Disposal_3_Day_High", Integer, nullable=True)
    Disposal_3_Day_Low = Column("Disposal_3_Day_Low", Integer, nullable=True)
    Disposal_3_Day_Variance = Column("Disposal_3_Day_Variance", Float, nullable=True)
    
    Disposal_3_Twilight_Avg = Column("Disposal_3_Twilight_Avg", Float, nullable=True)
    Disposal_3_Twilight_Median = Column("Disposal_3_Twilight_Median", Float, nullable=True)
    Disposal_3_Twilight_High = Column("Disposal_3_Twilight_High", Integer, nullable=True)
    Disposal_3_Twilight_Low = Column("Disposal_3_Twilight_Low", Integer, nullable=True)
    Disposal_3_Twilight_Variance = Column("Disposal_3_Twilight_Variance", Float, nullable=True)
    
    Disposal_3_Night_Avg = Column("Disposal_3_Night_Avg", Float, nullable=True)
    Disposal_3_Night_Median = Column("Disposal_3_Night_Median", Float, nullable=True)
    Disposal_3_Night_High = Column("Disposal_3_Night_High", Integer, nullable=True)
    Disposal_3_Night_Low = Column("Disposal_3_Night_Low", Integer, nullable=True)
    Disposal_3_Night_Variance = Column("Disposal_3_Night_Variance", Float, nullable=True)
    
    Disposal_3_Home_Avg = Column("Disposal_3_Home_Avg", Float, nullable=True)
    Disposal_3_Home_Median = Column("Disposal_3_Home_Median", Float, nullable=True)
    Disposal_3_Home_High = Column("Disposal_3_Home_High", Integer, nullable=True)
    Disposal_3_Home_Low = Column("Disposal_3_Home_Low", Integer, nullable=True)
    Disposal_3_Home_Variance = Column("Disposal_3_Home_Variance", Float, nullable=True)
    
    Disposal_3_Away_Avg = Column("Disposal_3_Away_Avg", Float, nullable=True)
    Disposal_3_Away_Median = Column("Disposal_3_Away_Median", Float, nullable=True)
    Disposal_3_Away_High = Column("Disposal_3_Away_High", Integer, nullable=True)
    Disposal_3_Away_Low = Column("Disposal_3_Away_Low", Integer, nullable=True)
    Disposal_3_Away_Variance = Column("Disposal_3_Away_Variance", Float, nullable=True)
    
    Disposal_6_Avg = Column("Disposal_6_Avg", Float, nullable=True)
    Disposal_6_Median = Column("Disposal_6_Median", Float, nullable=True)
    Disposal_6_High = Column("Disposal_6_High", Integer, nullable=True)
    Disposal_6_Low = Column("Disposal_6_Low", Integer, nullable=True)
    Disposal_6_Variance = Column("Disposal_6_Variance", Float, nullable=True)
    
    Disposal_6_Dry_Avg = Column("Disposal_6_Dry_Avg", Float, nullable=True)
    Disposal_6_Dry_Median = Column("Disposal_6_Dry_Median", Float, nullable=True)
    Disposal_6_Dry_High = Column("Disposal_6_Dry_High", Integer, nullable=True)
    Disposal_6_Dry_Low = Column("Disposal_6_Dry_Low", Integer, nullable=True)
    Disposal_6_Dry_Variance = Column("Disposal_6_Dry_Variance", Float, nullable=True)
    
    Disposal_6_Wet_Avg = Column("Disposal_6_Wet_Avg", Float, nullable=True)
    Disposal_6_Wet_Median = Column("Disposal_6_Wet_Median", Float, nullable=True)
    Disposal_6_Wet_High = Column("Disposal_6_Wet_High", Integer, nullable=True)
    Disposal_6_Wet_Low = Column("Disposal_6_Wet_Low", Integer, nullable=True)
    Disposal_6_Wet_Variance = Column("Disposal_6_Wet_Variance", Float, nullable=True)
    
    Disposal_6_Day_Avg = Column("Disposal_6_Day_Avg", Float, nullable=True)
    Disposal_6_Day_Median = Column("Disposal_6_Day_Median", Float, nullable=True)
    Disposal_6_Day_High = Column("Disposal_6_Day_High", Integer, nullable=True)
    Disposal_6_Day_Low = Column("Disposal_6_Day_Low", Integer, nullable=True)
    Disposal_6_Day_Variance = Column("Disposal_6_Day_Variance", Float, nullable=True)
    
    Disposal_6_Twilight_Avg = Column("Disposal_6_Twilight_Avg", Float, nullable=True)
    Disposal_6_Twilight_Median = Column("Disposal_6_Twilight_Median", Float, nullable=True)
    Disposal_6_Twilight_High = Column("Disposal_6_Twilight_High", Integer, nullable=True)
    Disposal_6_Twilight_Low = Column("Disposal_6_Twilight_Low", Integer, nullable=True)
    Disposal_6_Twilight_Variance = Column("Disposal_6_Twilight_Variance", Float, nullable=True)
    
    Disposal_6_Night_Avg = Column("Disposal_6_Night_Avg", Float, nullable=True)
    Disposal_6_Night_Median = Column("Disposal_6_Night_Median", Float, nullable=True)
    Disposal_6_Night_High = Column("Disposal_6_Night_High", Integer, nullable=True)
    Disposal_6_Night_Low = Column("Disposal_6_Night_Low", Integer, nullable=True)
    Disposal_6_Night_Variance = Column("Disposal_6_Night_Variance", Float, nullable=True)
    
    Disposal_6_Home_Avg = Column("Disposal_6_Home_Avg", Float, nullable=True)
    Disposal_6_Home_Median = Column("Disposal_6_Home_Median", Float, nullable=True)
    Disposal_6_Home_High = Column("Disposal_6_Home_High", Integer, nullable=True)
    Disposal_6_Home_Low = Column("Disposal_6_Home_Low", Integer, nullable=True)
    Disposal_6_Home_Variance = Column("Disposal_6_Home_Variance", Float, nullable=True)
    
    Disposal_6_Away_Avg = Column("Disposal_6_Away_Avg", Float, nullable=True)
    Disposal_6_Away_Median = Column("Disposal_6_Away_Median", Float, nullable=True)
    Disposal_6_Away_High = Column("Disposal_6_Away_High", Integer, nullable=True)
    Disposal_6_Away_Low = Column("Disposal_6_Away_Low", Integer, nullable=True)
    Disposal_6_Away_Variance = Column("Disposal_6_Away_Variance", Float, nullable=True)
    
    Disposal_10_Avg = Column("Disposal_10_Avg", Float, nullable=True)
    Disposal_10_Median = Column("Disposal_10_Median", Float, nullable=True)
    Disposal_10_High = Column("Disposal_10_High", Integer, nullable=True)
    Disposal_10_Low = Column("Disposal_10_Low", Integer, nullable=True)
    Disposal_10_Variance = Column("Disposal_10_Variance", Float, nullable=True)
    
    Disposal_10_Dry_Avg = Column("Disposal_10_Dry_Avg", Float, nullable=True)
    Disposal_10_Dry_Median = Column("Disposal_10_Dry_Median", Float, nullable=True)
    Disposal_10_Dry_High = Column("Disposal_10_Dry_High", Integer, nullable=True)
    Disposal_10_Dry_Low = Column("Disposal_10_Dry_Low", Integer, nullable=True)
    Disposal_10_Dry_Variance = Column("Disposal_10_Dry_Variance", Float, nullable=True)
    
    Disposal_10_Wet_Avg = Column("Disposal_10_Wet_Avg", Float, nullable=True)
    Disposal_10_Wet_Median = Column("Disposal_10_Wet_Median", Float, nullable=True)
    Disposal_10_Wet_High = Column("Disposal_10_Wet_High", Integer, nullable=True)
    Disposal_10_Wet_Low = Column("Disposal_10_Wet_Low", Integer, nullable=True)
    Disposal_10_Wet_Variance = Column("Disposal_10_Wet_Variance", Float, nullable=True)
    
    Disposal_10_Day_Avg = Column("Disposal_10_Day_Avg", Float, nullable=True)
    Disposal_10_Day_Median = Column("Disposal_10_Day_Median", Float, nullable=True)
    Disposal_10_Day_High = Column("Disposal_10_Day_High", Integer, nullable=True)
    Disposal_10_Day_Low = Column("Disposal_10_Day_Low", Integer, nullable=True)
    Disposal_10_Day_Variance = Column("Disposal_10_Day_Variance", Float, nullable=True)
    
    Disposal_10_Twilight_Avg = Column("Disposal_10_Twilight_Avg", Float, nullable=True)
    Disposal_10_Twilight_Median = Column("Disposal_10_Twilight_Median", Float, nullable=True)
    Disposal_10_Twilight_High = Column("Disposal_10_Twilight_High", Integer, nullable=True)
    Disposal_10_Twilight_Low = Column("Disposal_10_Twilight_Low", Integer, nullable=True)
    Disposal_10_Twilight_Variance = Column("Disposal_10_Twilight_Variance", Float, nullable=True)
    
    Disposal_10_Night_Avg = Column("Disposal_10_Night_Avg", Float, nullable=True)
    Disposal_10_Night_Median = Column("Disposal_10_Night_Median", Float, nullable=True)
    Disposal_10_Night_High = Column("Disposal_10_Night_High", Integer, nullable=True)
    Disposal_10_Night_Low = Column("Disposal_10_Night_Low", Integer, nullable=True)
    Disposal_10_Night_Variance = Column("Disposal_10_Night_Variance", Float, nullable=True)
    
    Disposal_10_Home_Avg = Column("Disposal_10_Home_Avg", Float, nullable=True)
    Disposal_10_Home_Median = Column("Disposal_10_Home_Median", Float, nullable=True)
    Disposal_10_Home_High = Column("Disposal_10_Home_High", Integer, nullable=True)
    Disposal_10_Home_Low = Column("Disposal_10_Home_Low", Integer, nullable=True)
    Disposal_10_Home_Variance = Column("Disposal_10_Home_Variance", Float, nullable=True)
    
    Disposal_10_Away_Avg = Column("Disposal_10_Away_Avg", Float, nullable=True)
    Disposal_10_Away_Median = Column("Disposal_10_Away_Median", Float, nullable=True)
    Disposal_10_Away_High = Column("Disposal_10_Away_High", Integer, nullable=True)
    Disposal_10_Away_Low = Column("Disposal_10_Away_Low", Integer, nullable=True)
    Disposal_10_Away_Variance = Column("Disposal_10_Away_Variance", Float, nullable=True)
    
    Disposal_22_Avg = Column("Disposal_22_Avg", Float, nullable=True)
    Disposal_22_Median = Column("Disposal_22_Median", Float, nullable=True)
    Disposal_22_High = Column("Disposal_22_High", Integer, nullable=True)
    Disposal_22_Low = Column("Disposal_22_Low", Integer, nullable=True)
    Disposal_22_Variance = Column("Disposal_22_Variance", Float, nullable=True)

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

class PlayerNickname(Base):
    __tablename__ = "player_nicknames"

    Nickname = Column("Nickname", String, primary_key=True)
    Player = Column("Player", String, primary_key=True)

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
    
class TeamPrecompute(Base):
    __tablename__ = "team_precompute_latest"

    Team = Column("Team", String, primary_key=True,nullable=True)

    season_avg_score = Column("season_avg_score", Integer, nullable=True)
    season_min_score = Column("season_min_score", Integer, nullable=True)
    season_max_score = Column("season_max_score", Integer, nullable=True)
    season_avg_home_score = Column("season_avg_home_score", Integer, nullable=True)
    season_min_home_score = Column("season_min_home_score", Integer, nullable=True) 
    season_max_home_score = Column("season_max_home_score", Integer, nullable=True)
    season_avg_away_score = Column("season_avg_away_score", Integer, nullable=True)
    season_min_away_score = Column("season_min_away_score", Integer, nullable=True)
    season_max_away_score = Column("season_max_away_score", Integer, nullable=True)

    season_avg_disposals = Column("season_avg_disposals", Integer, nullable=True)
    season_min_disposals = Column("season_min_disposals", Integer, nullable=True)
    season_max_disposals = Column("season_max_disposals", Integer, nullable=True)
    season_avg_home_disposals = Column("season_avg_home_disposals", Integer, nullable=True)
    season_min_home_disposals = Column("season_min_home_disposals", Integer, nullable=True)
    season_max_home_disposals = Column("season_max_home_disposals", Integer, nullable=True)
    season_avg_away_disposals = Column("season_avg_away_disposals", Integer, nullable=True)
    season_min_away_disposals = Column("season_min_away_disposals", Integer, nullable=True)
    season_max_away_disposals = Column("season_max_away_disposals", Integer, nullable=True)

    season_avg_goals = Column("season_avg_goals", Integer, nullable=True)
    season_min_goals = Column("season_min_goals", Integer, nullable=True)
    season_max_goals = Column("season_max_goals", Integer, nullable=True)
    season_avg_home_goals = Column("season_avg_home_goals", Integer, nullable=True)
    season_min_home_goals = Column("season_min_home_goals", Integer, nullable=True)
    season_max_home_goals = Column("season_max_home_goals", Integer, nullable=True)
    season_avg_away_goals = Column("season_avg_away_goals", Integer, nullable=True)
    season_min_away_goals = Column("season_min_away_goals", Integer, nullable=True)
    season_max_away_goals = Column("season_max_away_goals", Integer, nullable=True)

    season_avg_clearances = Column("season_avg_clearances", Integer, nullable=True)
    season_min_clearances = Column("season_min_clearances", Integer, nullable=True)
    season_max_clearances = Column("season_max_clearances", Integer, nullable=True)
    season_avg_home_clearances = Column("season_avg_home_clearances", Integer, nullable=True)
    season_min_home_clearances = Column("season_min_home_clearances", Integer, nullable=True)
    season_max_home_clearances = Column("season_max_home_clearances", Integer, nullable=True)
    season_avg_away_clearances = Column("season_avg_away_clearances", Integer, nullable=True)
    season_min_away_clearances = Column("season_min_away_clearances", Integer, nullable=True)
    season_max_away_clearances = Column("season_max_away_clearances", Integer, nullable=True)

    season_avg_tackles = Column("season_avg_tackles", Integer, nullable=True)
    season_min_tackles = Column("season_min_tackles", Integer, nullable=True)
    season_max_tackles = Column("season_max_tackles", Integer, nullable=True)
    season_avg_home_tackles = Column("season_avg_home_tackles", Integer, nullable=True)
    season_min_home_tackles = Column("season_min_home_tackles", Integer, nullable=True)
    season_max_home_tackles = Column("season_max_home_tackles", Integer, nullable=True)
    season_avg_away_tackles = Column("season_avg_away_tackles", Integer, nullable=True)
    season_min_away_tackles = Column("season_min_away_tackles", Integer, nullable=True)
    season_max_away_tackles = Column("season_max_away_tackles", Integer, nullable=True)

    season_avg_inside50 = Column("season_avg_inside50", Integer, nullable=True)
    season_min_inside50 = Column("season_min_inside50", Integer, nullable=True)
    season_max_inside50 = Column("season_max_inside50", Integer, nullable=True)
    season_avg_home_inside50 = Column("season_avg_home_inside50", Integer, nullable=True)
    season_min_home_inside50 = Column("season_min_home_inside50", Integer, nullable=True)
    season_max_home_inside50 = Column("season_max_home_inside50", Integer, nullable=True)
    season_avg_away_inside50 = Column("season_avg_away_inside50", Integer, nullable=True)
    season_min_away_inside50 = Column("season_min_away_inside50", Integer, nullable=True)
    season_max_away_inside50 = Column("season_max_away_inside50", Integer, nullable=True)

    season_avg_turnovers = Column("season_avg_turnovers", Integer, nullable=True)
    season_min_turnovers = Column("season_min_turnovers", Integer, nullable=True)
    season_max_turnovers = Column("season_max_turnovers", Integer, nullable=True)
    season_avg_home_turnovers = Column("season_avg_home_turnovers", Integer, nullable=True)
    season_min_home_turnovers = Column("season_min_home_turnovers", Integer, nullable=True)
    season_max_home_turnovers = Column("season_max_home_turnovers", Integer, nullable=True)
    season_avg_away_turnovers = Column("season_avg_away_turnovers", Integer, nullable=True)
    season_min_away_turnovers = Column("season_min_away_turnovers", Integer, nullable=True)
    season_max_away_turnovers = Column("season_max_away_turnovers", Integer, nullable=True)

    season_avg_free_kicks = Column("season_avg_free_kicks", Integer, nullable=True)
    season_min_free_kicks = Column("season_min_free_kicks", Integer, nullable=True)
    season_max_free_kicks = Column("season_max_free_kicks", Integer, nullable=True)
    season_avg_home_free_kicks = Column("season_avg_home_free_kicks", Integer, nullable=True)
    season_min_home_free_kicks = Column("season_min_home_free_kicks", Integer, nullable=True)
    season_max_home_free_kicks = Column("season_max_home_free_kicks", Integer, nullable=True)
    season_avg_away_free_kicks = Column("season_avg_away_free_kicks", Integer, nullable=True)
    season_min_away_free_kicks = Column("season_min_away_free_kicks", Integer, nullable=True)
    season_max_away_free_kicks = Column("season_max_away_free_kicks", Integer, nullable=True)

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
