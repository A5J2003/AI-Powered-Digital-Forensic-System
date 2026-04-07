import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface InputTypeCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  onClick: () => void;
  index: number;
  gradient: string;
}

const InputTypeCard = ({
  icon: Icon,
  title,
  description,
  onClick,
  index,
  gradient,
}: InputTypeCardProps) => {
  return (
    <motion.button
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index, duration: 0.4 }}
      whileHover={{ scale: 1.02, y: -4 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="group relative w-full text-left"
    >
      {/* Glow effect on hover */}
      <div className={`absolute inset-0 ${gradient} rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-300`} />
      
      <div className="relative bg-card/60 backdrop-blur-sm border border-border/50 rounded-2xl p-6 transition-all duration-300 group-hover:border-primary/30 group-hover:bg-card/80">
        <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl ${gradient} mb-4`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        
        <h3 className="text-lg font-semibold text-foreground mb-1">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
        
        {/* Arrow indicator */}
        <div className="absolute right-6 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
          <motion.span
            animate={{ x: [0, 4, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="text-primary text-xl"
          >
            →
          </motion.span>
        </div>
      </div>
    </motion.button>
  );
};

export default InputTypeCard;
