// src/utils/writer.ts
import { Page, Writer, PageWithRelations } from '@/services/api';

export function getWriterDisplay(page: Page | PageWithRelations, writers: Writer[]): string {
    if ('writer' in page && page.writer) {
        return page.writer.name;
    }
    if (page.writer_id) {
        const writer = writers.find(w => w.id === page.writer_id);
        return writer?.name || 'Unknown Writer';
    }
    return 'Unassigned';
}