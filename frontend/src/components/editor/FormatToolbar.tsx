// src/components/editor/FormatToolbar.tsx
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
    Bold,
    Italic,
    Underline,
    AlignLeft,
    AlignCenter,
    AlignRight,
    List,
} from "lucide-react";

interface FormatToolbarProps {
    onFormatClick: (format: string) => void;
}

export function FormatToolbar({ onFormatClick }: FormatToolbarProps) {
    const formatButtons = [
        { icon: Bold, format: 'bold', tooltip: 'Bold' },
        { icon: Italic, format: 'italic', tooltip: 'Italic' },
        { icon: Underline, format: 'underline', tooltip: 'Underline' },
        { icon: AlignLeft, format: 'alignLeft', tooltip: 'Align Left' },
        { icon: AlignCenter, format: 'alignCenter', tooltip: 'Align Center' },
        { icon: AlignRight, format: 'alignRight', tooltip: 'Align Right' },
        { icon: List, format: 'list', tooltip: 'Bullet List' },
    ];

    return (
        <div className="mb-4 flex items-center space-x-1 border-b pb-4">
            {formatButtons.map((button) => (
                <Tooltip key={button.format}>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onFormatClick(button.format)}
                        >
                            <button.icon className="w-4 h-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent>{button.tooltip}</TooltipContent>
                </Tooltip>
            ))}
        </div>
    );
}

export default FormatToolbar;