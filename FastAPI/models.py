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

    Prob_20_Disp_Dry = Column("Prob_20_Disp_Dry", Float, nullable=True)
    Prob_20_Disp_Wet = Column("Prob_20_Disp_Wet", Float, nullable=True)
    Prob_25_Disp_Dry = Column("Prob_25_Disp_Dry", Float, nullable=True)
    Prob_25_Disp_Wet = Column("Prob_25_Disp_Wet", Float, nullable=True)
    Prob_30_Disp_Dry = Column("Prob_30_Disp_Dry", Float, nullable=True)
    Prob_30_Disp_Wet = Column("Prob_30_Disp_Wet", Float, nullable=True)

    Disposal_Season_Avg = Column("Disposal_Season_Avg", Float, nullable=True)
    Disposal_Season_Median = Column("Disposal_Season_Median", Float, nullable=True)
    Disposal_Season_High = Column("Disposal_Season_High", Integer, nullable=True)
    Disposal_Season_Low = Column("Disposal_Season_Low", Integer, nullable=True)
    Disposal_Season_Variance = Column("Disposal_Season_Variance", Float, nullable=True)

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
